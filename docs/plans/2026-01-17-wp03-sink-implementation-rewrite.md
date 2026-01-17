# WP-03: Sink Implementation Rewrite

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite all sink implementations to use batch write signature with ArtifactDescriptor return, providing content hashing for audit integrity.

**Architecture:** Each sink (CSV, JSON, Database) changes from per-row `write(row) -> None` to batch `write(rows) -> ArtifactDescriptor`. Content hashing uses SHA-256. File sinks hash the written file; database sink hashes the canonical JSON payload before INSERT. All sinks gain `plugin_version` and explicit lifecycle hooks.

**Tech Stack:** Python 3.12, hashlib (SHA-256), dataclasses, SQLAlchemy Core

---

## Implementation Rules

> **HIGH PROFILE SYSTEM - STRICT COMPLIANCE REQUIRED**

1. **CLAUDE.md Compliance:** All code must comply with CLAUDE.md directives. No defensive programming patterns. No backwards compatibility shims. No legacy code retention.

2. **Auditability is Priority:** Every sink write MUST return an `ArtifactDescriptor` with:
   - `content_hash`: SHA-256 proving what was written (REQUIRED, non-empty)
   - `size_bytes`: Exact byte count (REQUIRED, > 0 for non-empty writes)
   - `path_or_uri`: Where the artifact lives

3. **Content Hash Semantics:**
   | Sink Type | What Gets Hashed | Why |
   |-----------|------------------|-----|
   | **CSV/JSON** | SHA-256 of file contents after write | Proves exact bytes on disk |
   | **Database** | SHA-256 of canonical JSON payload BEFORE insert | Proves intent (DB may transform) |

4. **Deviation Reporting:** Report ALL deviations from best practice, no matter how minor. Flag with `# DEVIATION:` comment explaining why.

5. **No Silent Failures:** If a write fails, raise an exception. Never return a fake ArtifactDescriptor.

6. **TDD Strictly Enforced:** Write failing test → Run to verify fail → Implement → Run to verify pass → Commit. No shortcuts.

---

## Task 1: CSVSink Batch Write Implementation

**Files:**
- Modify: `src/elspeth/plugins/sinks/csv_sink.py`
- Test: `tests/plugins/sinks/test_csv_sink.py`

**Step 1: Write failing tests for batch write signature**

Add to `tests/plugins/sinks/test_csv_sink.py`:

```python
# Add these imports at the top if not present
import hashlib
from typing import Any

# Add to TestCSVSink class

    def test_batch_write_returns_artifact_descriptor(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """write() returns ArtifactDescriptor with content hash."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        artifact = sink.write([{"id": "1", "name": "alice"}], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.artifact_type == "file"
        assert artifact.content_hash  # Non-empty
        assert artifact.size_bytes > 0

    def test_batch_write_content_hash_is_sha256(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """content_hash is SHA-256 of file contents."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        artifact = sink.write([{"id": "1", "name": "alice"}], ctx)
        sink.close()

        # Compute expected hash from file
        file_content = output_file.read_bytes()
        expected_hash = hashlib.sha256(file_content).hexdigest()

        assert artifact.content_hash == expected_hash

    def test_batch_write_multiple_rows(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Batch write handles multiple rows."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        rows = [
            {"id": "1", "name": "alice"},
            {"id": "2", "name": "bob"},
            {"id": "3", "name": "carol"},
        ]
        artifact = sink.write(rows, ctx)
        sink.close()

        assert artifact.size_bytes > 0

        # Verify all rows written
        with open(output_file) as f:
            reader = csv.DictReader(f)
            written_rows = list(reader)
        assert len(written_rows) == 3

    def test_batch_write_empty_list(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Batch write with empty list returns descriptor with zero size."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        artifact = sink.write([], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.size_bytes == 0
        # Empty write still has a hash (of empty content)
        assert artifact.content_hash == hashlib.sha256(b"").hexdigest()

    def test_has_plugin_version(self) -> None:
        """CSVSink has plugin_version attribute."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        sink = CSVSink({"path": "/tmp/test.csv"})
        assert hasattr(sink, "plugin_version")
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """CSVSink has determinism attribute."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.sinks.csv_sink import CSVSink

        sink = CSVSink({"path": "/tmp/test.csv"})
        assert sink.determinism == Determinism.IO_WRITE
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/sinks/test_csv_sink.py::TestCSVSink::test_batch_write_returns_artifact_descriptor -v`

Expected: FAIL with `TypeError: write() takes 3 positional arguments but got list`

**Step 3: Implement batch write for CSVSink**

Replace the entire `src/elspeth/plugins/sinks/csv_sink.py`:

```python
# src/elspeth/plugins/sinks/csv_sink.py
"""CSV sink plugin for ELSPETH.

Writes rows to CSV files with content hashing for audit integrity.
"""

import csv
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PathConfig
from elspeth.plugins.context import PluginContext


class CSVInputSchema(PluginSchema):
    """Dynamic schema - accepts any row structure."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern


class CSVSinkConfig(PathConfig):
    """Configuration for CSV sink plugin."""

    delimiter: str = ","
    encoding: str = "utf-8"


class CSVSink(BaseSink):
    """Write rows to a CSV file.

    Returns ArtifactDescriptor with SHA-256 content hash for audit integrity.

    Config options:
        path: Path to output CSV file (required)
        delimiter: Field delimiter (default: ",")
        encoding: File encoding (default: "utf-8")
    """

    name = "csv"
    input_schema = CSVInputSchema
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = CSVSinkConfig.from_dict(config)
        self._path = cfg.resolved_path()
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding

        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: Sequence[str] | None = None

    def write(
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> ArtifactDescriptor:
        """Write a batch of rows to the CSV file.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash (SHA-256) and size_bytes
        """
        if not rows:
            # Empty batch - return descriptor for empty content
            return ArtifactDescriptor.for_file(
                path=str(self._path),
                content_hash=hashlib.sha256(b"").hexdigest(),
                size_bytes=0,
            )

        # Lazy initialization - discover fieldnames from first row
        if self._file is None:
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

        # Write all rows in batch
        for row in rows:
            self._writer.writerow(row)  # type: ignore[union-attr]

        # Flush to ensure content is on disk for hashing
        self._file.flush()

        # Compute content hash from file
        content_hash = self._compute_file_hash()
        size_bytes = self._path.stat().st_size

        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    def _compute_file_hash(self) -> str:
        """Compute SHA-256 hash of the file contents."""
        sha256 = hashlib.sha256()
        with open(self._path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def flush(self) -> None:
        """Flush buffered data to disk."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    # === Lifecycle Hooks ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before processing begins."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called after processing completes."""
        pass
```

**Step 4: Run new tests to verify they pass**

Run: `pytest tests/plugins/sinks/test_csv_sink.py::TestCSVSink::test_batch_write_returns_artifact_descriptor tests/plugins/sinks/test_csv_sink.py::TestCSVSink::test_batch_write_content_hash_is_sha256 tests/plugins/sinks/test_csv_sink.py::TestCSVSink::test_batch_write_multiple_rows tests/plugins/sinks/test_csv_sink.py::TestCSVSink::test_batch_write_empty_list tests/plugins/sinks/test_csv_sink.py::TestCSVSink::test_has_plugin_version tests/plugins/sinks/test_csv_sink.py::TestCSVSink::test_has_determinism -v`

Expected: All PASS

**Step 5: Update old tests to use batch signature**

Update the existing tests in `tests/plugins/sinks/test_csv_sink.py` to use batch signature:

```python
    def test_write_creates_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """write() creates CSV file with headers."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        sink.write([{"id": "1", "name": "alice"}], ctx)  # Changed to batch
        sink.flush()
        sink.close()

        assert output_file.exists()
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["id"] == "1"
        assert rows[0]["name"] == "alice"

    def test_write_multiple_rows(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Multiple writes append to CSV."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        # Changed to batch writes
        sink.write([{"id": "1", "name": "alice"}], ctx)
        sink.write([{"id": "2", "name": "bob"}], ctx)
        sink.write([{"id": "3", "name": "carol"}], ctx)
        sink.flush()
        sink.close()

        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[2]["name"] == "carol"

    def test_custom_delimiter(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Support custom delimiter."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "delimiter": ";"})

        sink.write([{"id": "1", "name": "alice"}], ctx)  # Changed to batch
        sink.flush()
        sink.close()

        content = output_file.read_text()
        assert ";" in content
        assert "," not in content.replace(",", "")

    def test_close_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        sink.write([{"id": "1"}], ctx)  # Changed to batch
        sink.close()
        sink.close()  # Should not raise

    def test_flush_before_close(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Data is available after flush, before close."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        sink.write([{"id": "1"}], ctx)  # Changed to batch
        sink.flush()

        content = output_file.read_text()
        assert "id" in content
        assert "1" in content

        sink.close()
```

**Step 6: Run all CSV sink tests**

Run: `pytest tests/plugins/sinks/test_csv_sink.py -v`

Expected: All PASS

**Step 7: Commit**

```bash
git add src/elspeth/plugins/sinks/csv_sink.py tests/plugins/sinks/test_csv_sink.py
git commit -m "feat(csv-sink): implement batch write with ArtifactDescriptor

- Change write(row) -> write(rows) batch signature
- Return ArtifactDescriptor with SHA-256 content hash
- Add plugin_version = 1.0.0
- Add on_start() and on_complete() lifecycle hooks
- Update all tests to use batch signature

Part of WP-03: Sink Implementation Rewrite"
```

---

## Task 2: JSONSink Batch Write Implementation

**Files:**
- Modify: `src/elspeth/plugins/sinks/json_sink.py`
- Test: `tests/plugins/sinks/test_json_sink.py`

**Step 1: Write failing tests for batch write signature**

Add to `tests/plugins/sinks/test_json_sink.py`:

```python
# Add these imports at the top if not present
import hashlib
from typing import Any

# Add to TestJSONSink class

    def test_batch_write_returns_artifact_descriptor(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """write() returns ArtifactDescriptor with content hash."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        artifact = sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.artifact_type == "file"
        assert artifact.content_hash  # Non-empty
        assert artifact.size_bytes > 0

    def test_batch_write_content_hash_is_sha256(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """content_hash is SHA-256 of file contents."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        artifact = sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        file_content = output_file.read_bytes()
        expected_hash = hashlib.sha256(file_content).hexdigest()

        assert artifact.content_hash == expected_hash

    def test_batch_write_jsonl_content_hash(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """JSONL format also returns correct content hash."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = JSONSink({"path": str(output_file)})

        artifact = sink.write([{"id": 1}, {"id": 2}], ctx)
        sink.close()

        file_content = output_file.read_bytes()
        expected_hash = hashlib.sha256(file_content).hexdigest()

        assert artifact.content_hash == expected_hash

    def test_batch_write_empty_list(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Batch write with empty list returns descriptor with zero size."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        artifact = sink.write([], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.size_bytes == 0
        assert artifact.content_hash == hashlib.sha256(b"").hexdigest()

    def test_has_plugin_version(self) -> None:
        """JSONSink has plugin_version attribute."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        sink = JSONSink({"path": "/tmp/test.json"})
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """JSONSink has determinism attribute."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.sinks.json_sink import JSONSink

        sink = JSONSink({"path": "/tmp/test.json"})
        assert sink.determinism == Determinism.IO_WRITE
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/sinks/test_json_sink.py::TestJSONSink::test_batch_write_returns_artifact_descriptor -v`

Expected: FAIL with `TypeError`

**Step 3: Implement batch write for JSONSink**

Replace the entire `src/elspeth/plugins/sinks/json_sink.py`:

```python
# src/elspeth/plugins/sinks/json_sink.py
"""JSON sink plugin for ELSPETH.

Writes rows to JSON files with content hashing for audit integrity.
Supports JSON array and JSONL formats.
"""

import hashlib
import json
from pathlib import Path
from typing import IO, Any, Literal

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PathConfig
from elspeth.plugins.context import PluginContext


class JSONInputSchema(PluginSchema):
    """Dynamic schema - accepts any row structure."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern


class JSONSinkConfig(PathConfig):
    """Configuration for JSON sink plugin."""

    format: Literal["json", "jsonl"] | None = None
    indent: int | None = None
    encoding: str = "utf-8"


class JSONSink(BaseSink):
    """Write rows to a JSON file.

    Returns ArtifactDescriptor with SHA-256 content hash for audit integrity.

    Config options:
        path: Path to output JSON file (required)
        format: "json" (array) or "jsonl" (lines). Auto-detected from extension.
        indent: Indentation for pretty-printing (default: None for compact)
        encoding: File encoding (default: "utf-8")
    """

    name = "json"
    input_schema = JSONInputSchema
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = JSONSinkConfig.from_dict(config)
        self._path = cfg.resolved_path()
        self._encoding = cfg.encoding
        self._indent = cfg.indent

        # Auto-detect format from extension if not specified
        fmt = cfg.format
        if fmt is None:
            fmt = "jsonl" if self._path.suffix == ".jsonl" else "json"
        self._format = fmt

        self._file: IO[str] | None = None
        self._rows: list[dict[str, Any]] = []  # Buffer for json array format

    def write(
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> ArtifactDescriptor:
        """Write a batch of rows to the JSON file.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash (SHA-256) and size_bytes
        """
        if not rows:
            # Empty batch - return descriptor for empty content
            return ArtifactDescriptor.for_file(
                path=str(self._path),
                content_hash=hashlib.sha256(b"").hexdigest(),
                size_bytes=0,
            )

        if self._format == "jsonl":
            return self._write_jsonl_batch(rows)
        else:
            return self._write_json_batch(rows)

    def _write_jsonl_batch(
        self, rows: list[dict[str, Any]]
    ) -> ArtifactDescriptor:
        """Write rows as JSONL (one object per line)."""
        if self._file is None:
            self._file = open(self._path, "w", encoding=self._encoding)

        for row in rows:
            json.dump(row, self._file)
            self._file.write("\n")

        self._file.flush()

        content_hash = self._compute_file_hash()
        size_bytes = self._path.stat().st_size

        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    def _write_json_batch(
        self, rows: list[dict[str, Any]]
    ) -> ArtifactDescriptor:
        """Write rows as JSON array."""
        self._rows.extend(rows)

        # Write all accumulated rows as array
        if self._file is None:
            self._file = open(self._path, "w", encoding=self._encoding)

        self._file.seek(0)
        self._file.truncate()
        json.dump(self._rows, self._file, indent=self._indent)
        self._file.flush()

        content_hash = self._compute_file_hash()
        size_bytes = self._path.stat().st_size

        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    def _compute_file_hash(self) -> str:
        """Compute SHA-256 hash of the file contents."""
        sha256 = hashlib.sha256()
        with open(self._path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def flush(self) -> None:
        """Flush buffered data to disk."""
        if self._format == "json" and self._rows:
            if self._file is None:
                self._file = open(self._path, "w", encoding=self._encoding)
            self._file.seek(0)
            self._file.truncate()
            json.dump(self._rows, self._file, indent=self._indent)

        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._format == "json" and self._rows and self._file is None:
            self.flush()

        if self._file is not None:
            self._file.close()
            self._file = None
            self._rows = []

    # === Lifecycle Hooks ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before processing begins."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called after processing completes."""
        pass
```

**Step 4: Run new tests to verify they pass**

Run: `pytest tests/plugins/sinks/test_json_sink.py::TestJSONSink::test_batch_write_returns_artifact_descriptor tests/plugins/sinks/test_json_sink.py::TestJSONSink::test_batch_write_content_hash_is_sha256 tests/plugins/sinks/test_json_sink.py::TestJSONSink::test_batch_write_jsonl_content_hash tests/plugins/sinks/test_json_sink.py::TestJSONSink::test_batch_write_empty_list tests/plugins/sinks/test_json_sink.py::TestJSONSink::test_has_plugin_version tests/plugins/sinks/test_json_sink.py::TestJSONSink::test_has_determinism -v`

Expected: All PASS

**Step 5: Update old tests to use batch signature**

Update the existing tests in `tests/plugins/sinks/test_json_sink.py`:

```python
    def test_write_json_array(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Write rows as JSON array."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "format": "json"})

        sink.write([{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}], ctx)
        sink.flush()
        sink.close()

        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "alice"

    def test_write_jsonl(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Write rows as JSONL (one per line)."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = JSONSink({"path": str(output_file), "format": "jsonl"})

        sink.write([{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}], ctx)
        sink.flush()
        sink.close()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["name"] == "alice"
        assert json.loads(lines[1])["name"] == "bob"

    def test_auto_detect_format_from_extension(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Auto-detect JSONL format from .jsonl extension."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = JSONSink({"path": str(output_file)})

        sink.write([{"id": 1}], ctx)
        sink.flush()
        sink.close()

        content = output_file.read_text().strip()
        data = json.loads(content)
        assert data == {"id": 1}

    def test_json_extension_defaults_to_array(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Auto-detect JSON array format from .json extension."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        sink.write([{"id": 1}], ctx)
        sink.flush()
        sink.close()

        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert data == [{"id": 1}]

    def test_close_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        sink.write([{"id": 1}], ctx)
        sink.close()
        sink.close()

    def test_pretty_print_option(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Support pretty-printed JSON output."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "format": "json", "indent": 2})

        sink.write([{"id": 1}], ctx)
        sink.flush()
        sink.close()

        content = output_file.read_text()
        assert "\n" in content
        assert "  " in content
```

**Step 6: Run all JSON sink tests**

Run: `pytest tests/plugins/sinks/test_json_sink.py -v`

Expected: All PASS

**Step 7: Commit**

```bash
git add src/elspeth/plugins/sinks/json_sink.py tests/plugins/sinks/test_json_sink.py
git commit -m "feat(json-sink): implement batch write with ArtifactDescriptor

- Change write(row) -> write(rows) batch signature
- Return ArtifactDescriptor with SHA-256 content hash
- Support both JSON array and JSONL formats
- Add plugin_version = 1.0.0
- Add on_start() and on_complete() lifecycle hooks
- Update all tests to use batch signature

Part of WP-03: Sink Implementation Rewrite"
```

---

## Task 3: DatabaseSink Batch Write Implementation

**Files:**
- Modify: `src/elspeth/plugins/sinks/database_sink.py`
- Test: `tests/plugins/sinks/test_database_sink.py`

**Step 1: Write failing tests for batch write signature**

Add to `tests/plugins/sinks/test_database_sink.py`:

```python
# Add these imports at the top if not present
import hashlib
import json
from typing import Any

# Add to TestDatabaseSink class

    def test_batch_write_returns_artifact_descriptor(
        self, db_url: str, ctx: PluginContext
    ) -> None:
        """write() returns ArtifactDescriptor with content hash."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output"})

        artifact = sink.write([{"id": 1, "name": "alice"}], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.artifact_type == "database"
        assert artifact.content_hash  # Non-empty
        assert artifact.size_bytes > 0

    def test_batch_write_content_hash_is_payload_hash(
        self, db_url: str, ctx: PluginContext
    ) -> None:
        """content_hash is SHA-256 of canonical JSON payload BEFORE insert."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        rows = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
        sink = DatabaseSink({"url": db_url, "table": "output"})

        artifact = sink.write(rows, ctx)
        sink.close()

        # Hash should be of the canonical JSON payload
        # Note: We use sorted keys for canonical form
        payload_json = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        expected_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

        assert artifact.content_hash == expected_hash

    def test_batch_write_metadata_has_row_count(
        self, db_url: str, ctx: PluginContext
    ) -> None:
        """ArtifactDescriptor metadata includes row_count."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output"})

        artifact = sink.write([{"id": 1}, {"id": 2}, {"id": 3}], ctx)
        sink.close()

        assert artifact.metadata is not None
        assert artifact.metadata["row_count"] == 3

    def test_batch_write_empty_list(
        self, db_url: str, ctx: PluginContext
    ) -> None:
        """Batch write with empty list returns descriptor with zero size."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output"})

        artifact = sink.write([], ctx)
        sink.close()

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.size_bytes == 0
        # Empty payload hash
        empty_json = json.dumps([], sort_keys=True, separators=(",", ":"))
        assert artifact.content_hash == hashlib.sha256(empty_json.encode()).hexdigest()

    def test_has_plugin_version(self) -> None:
        """DatabaseSink has plugin_version attribute."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test"})
        assert sink.plugin_version == "1.0.0"

    def test_has_determinism(self) -> None:
        """DatabaseSink has determinism attribute."""
        from elspeth.contracts import Determinism
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test"})
        assert sink.determinism == Determinism.IO_WRITE
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/sinks/test_database_sink.py::TestDatabaseSink::test_batch_write_returns_artifact_descriptor -v`

Expected: FAIL with `TypeError`

**Step 3: Implement batch write for DatabaseSink**

Replace the entire `src/elspeth/plugins/sinks/database_sink.py`:

```python
# src/elspeth/plugins/sinks/database_sink.py
"""Database sink plugin for ELSPETH.

Writes rows to a database table with content hashing for audit integrity.
Uses SQLAlchemy Core for direct SQL control.
"""

import hashlib
import json
from typing import Any, Literal

from sqlalchemy import Column, MetaData, String, Table, create_engine, insert
from sqlalchemy.engine import Engine

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PluginConfig
from elspeth.plugins.context import PluginContext


class DatabaseInputSchema(PluginSchema):
    """Dynamic schema - accepts any row structure."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern


class DatabaseSinkConfig(PluginConfig):
    """Configuration for database sink plugin."""

    url: str
    table: str
    if_exists: Literal["append", "replace"] = "append"


class DatabaseSink(BaseSink):
    """Write rows to a database table.

    Returns ArtifactDescriptor with SHA-256 hash of canonical JSON payload
    BEFORE insert. This proves intent - the database may transform data
    (add timestamps, auto-increment IDs, etc.).

    Creates the table on first write, inferring columns from row keys.
    Uses SQLAlchemy Core for direct SQL control.

    Config options:
        url: Database connection URL (required)
        table: Table name (required)
        if_exists: "append" or "replace" (default: "append")
    """

    name = "database"
    input_schema = DatabaseInputSchema
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = DatabaseSinkConfig.from_dict(config)
        self._url = cfg.url
        self._table_name = cfg.table
        self._if_exists = cfg.if_exists

        self._engine: Engine | None = None
        self._table: Table | None = None
        self._metadata: MetaData | None = None

    def _ensure_table(self, row: dict[str, Any]) -> None:
        """Create table if it doesn't exist, inferring schema from row."""
        if self._engine is None:
            self._engine = create_engine(self._url)
            self._metadata = MetaData()

        if self._table is None:
            # Infer columns from first row (all as String for simplicity)
            columns = [Column(key, String) for key in row]
            assert self._metadata is not None
            self._table = Table(
                self._table_name,
                self._metadata,
                *columns,
            )
            self._metadata.create_all(self._engine, checkfirst=True)

    def write(
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> ArtifactDescriptor:
        """Write a batch of rows to the database.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash of canonical JSON payload
        """
        # Compute canonical JSON payload BEFORE any database operation
        # This proves intent regardless of what the database does
        payload_json = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        content_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        payload_size = len(payload_json.encode("utf-8"))

        if not rows:
            # Empty batch - return descriptor without DB operations
            return ArtifactDescriptor.for_database(
                url=self._url,
                table=self._table_name,
                content_hash=content_hash,
                payload_size=payload_size,
                row_count=0,
            )

        # Ensure table exists (infer from first row)
        self._ensure_table(rows[0])

        # Insert all rows
        assert self._engine is not None
        assert self._table is not None
        with self._engine.begin() as conn:
            conn.execute(insert(self._table), rows)

        return ArtifactDescriptor.for_database(
            url=self._url,
            table=self._table_name,
            content_hash=content_hash,
            payload_size=payload_size,
            row_count=len(rows),
        )

    def flush(self) -> None:
        """Flush any pending operations.

        For database sink, writes are immediate so this is a no-op.
        """
        pass

    def close(self) -> None:
        """Close database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._table = None
            self._metadata = None

    # === Lifecycle Hooks ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before processing begins."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called after processing completes."""
        pass
```

**Step 4: Run new tests to verify they pass**

Run: `pytest tests/plugins/sinks/test_database_sink.py::TestDatabaseSink::test_batch_write_returns_artifact_descriptor tests/plugins/sinks/test_database_sink.py::TestDatabaseSink::test_batch_write_content_hash_is_payload_hash tests/plugins/sinks/test_database_sink.py::TestDatabaseSink::test_batch_write_metadata_has_row_count tests/plugins/sinks/test_database_sink.py::TestDatabaseSink::test_batch_write_empty_list tests/plugins/sinks/test_database_sink.py::TestDatabaseSink::test_has_plugin_version tests/plugins/sinks/test_database_sink.py::TestDatabaseSink::test_has_determinism -v`

Expected: All PASS

**Step 5: Update old tests to use batch signature**

Update the existing tests in `tests/plugins/sinks/test_database_sink.py`:

```python
    def test_write_creates_table(self, db_url: str, ctx: PluginContext) -> None:
        """write() creates table and inserts rows."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output"})

        sink.write([{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}], ctx)
        sink.flush()
        sink.close()

        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)

        with engine.connect() as conn:
            rows = list(conn.execute(select(table)))

        assert len(rows) == 2
        assert rows[0][1] == "alice"

    def test_batch_insert(self, db_url: str, ctx: PluginContext) -> None:
        """Multiple rows in single batch."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output"})

        # Write 5 rows in single batch
        rows = [{"id": i, "value": f"val{i}"} for i in range(5)]
        sink.write(rows, ctx)
        sink.flush()
        sink.close()

        engine = create_engine(db_url)
        metadata = MetaData()
        table = Table("output", metadata, autoload_with=engine)

        with engine.connect() as conn:
            result_rows = list(conn.execute(select(table)))

        assert len(result_rows) == 5

    def test_close_is_idempotent(self, db_url: str, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": db_url, "table": "output"})

        sink.write([{"id": 1}], ctx)
        sink.close()
        sink.close()

    def test_memory_database(self, ctx: PluginContext) -> None:
        """Works with in-memory SQLite."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test"})

        sink.write([{"col": "value"}], ctx)
        sink.flush()
        sink.close()
```

**Step 6: Run all database sink tests**

Run: `pytest tests/plugins/sinks/test_database_sink.py -v`

Expected: All PASS

**Step 7: Commit**

```bash
git add src/elspeth/plugins/sinks/database_sink.py tests/plugins/sinks/test_database_sink.py
git commit -m "feat(database-sink): implement batch write with ArtifactDescriptor

- Change write(row) -> write(rows) batch signature
- Return ArtifactDescriptor with SHA-256 of canonical JSON payload
- Hash computed BEFORE insert (proves intent)
- Remove internal batching (now handled by caller)
- Add plugin_version = 1.0.0
- Add on_start() and on_complete() lifecycle hooks
- Update all tests to use batch signature

Part of WP-03: Sink Implementation Rewrite"
```

---

## Task 4: Type Checking and Integration Verification

**Files:**
- All modified sink files
- `src/elspeth/plugins/sinks/__init__.py`

**Step 1: Run mypy on all sink files**

Run: `mypy src/elspeth/plugins/sinks/csv_sink.py src/elspeth/plugins/sinks/json_sink.py src/elspeth/plugins/sinks/database_sink.py --strict`

Expected: All pass, no errors

**Step 2: Run all sink tests together**

Run: `pytest tests/plugins/sinks/ -v`

Expected: All PASS

**Step 3: Verify protocol conformance**

Run: `pytest tests/plugins/sinks/ -k "implements_protocol" -v`

Expected: All PASS - all sinks still implement SinkProtocol

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: verify WP-03 sink implementation rewrite complete

All sinks now:
- Use batch write(rows) -> ArtifactDescriptor signature
- Return SHA-256 content hash for audit integrity
- Have plugin_version = 1.0.0
- Have determinism = IO_WRITE (inherited from BaseSink)
- Implement on_start() and on_complete() lifecycle hooks

Mypy passes. All tests pass."
```

---

## Verification Checklist

- [ ] `CSVSink.write()` accepts `list[dict]` and returns `ArtifactDescriptor`
- [ ] `CSVSink` content_hash is SHA-256 of written file
- [ ] `CSVSink.plugin_version` == "1.0.0"
- [ ] `JSONSink.write()` accepts `list[dict]` and returns `ArtifactDescriptor`
- [ ] `JSONSink` content_hash is SHA-256 of written file (both formats)
- [ ] `JSONSink.plugin_version` == "1.0.0"
- [ ] `DatabaseSink.write()` accepts `list[dict]` and returns `ArtifactDescriptor`
- [ ] `DatabaseSink` content_hash is SHA-256 of canonical JSON payload BEFORE insert
- [ ] `DatabaseSink` metadata includes `row_count`
- [ ] `DatabaseSink.plugin_version` == "1.0.0"
- [ ] All sinks have `determinism == Determinism.IO_WRITE`
- [ ] All sinks have `on_start()` and `on_complete()` lifecycle hooks
- [ ] `mypy --strict` passes on all sink files
- [ ] `pytest tests/plugins/sinks/ -v` all pass
- [ ] No per-row `write(row)` calls remain in any test

---

## Deviations and Notes

### Content Hash for Empty Batches

Empty batch writes return a hash of empty content (`sha256(b"")` for files, `sha256(b"[]")` for database). This maintains the invariant that every ArtifactDescriptor has a valid content_hash, even for no-op writes.

### Database Sink Removed Internal Batching

The old DatabaseSink had internal buffering with `batch_size` config. This has been removed because:
1. The engine now controls batching via the batch `write(rows)` call
2. Internal buffering would complicate content hash computation
3. Simpler code is more auditable

The `batch_size` config option is removed from `DatabaseSinkConfig`.

### JSON Array Format Accumulation

For JSON array format, successive `write()` calls accumulate rows and rewrite the entire file. This ensures the content_hash always reflects the complete file state. For high-volume writes, JSONL format is recommended.
