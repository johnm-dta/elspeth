# Interface Unification: Audit-First Plugin Contracts

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the SinkAdapter band-aid by making SinkProtocol audit-aware from the start, then clean up unnecessary protocol duplication.

**Architecture:** The current design has `SinkProtocol` (row-wise, no artifact info) and `SinkLike` (batch, returns ArtifactDescriptor) with `SinkAdapter` bridging them. This refactor makes `SinkProtocol` the single source of truth by having it accept batches and return audit-required `ArtifactDescriptor`. Similar cleanup removes redundant `*Like` protocols.

**Tech Stack:** Python protocols, dataclasses, pytest

---

## Overview

| Issue | Fix |
|-------|-----|
| `SinkProtocol` vs `SinkLike` mismatch | Make `SinkProtocol` batch-based + return `ArtifactDescriptor` |
| `SinkAdapter` existence | Delete it - no longer needed |
| `TransformLike`, `GateLike`, `AggregationLike` duplication | Delete them - use full protocols |
| `TransformLike` name collision in orchestrator.py | Rename to explicit union type |

## Files Affected

**Delete entirely:**
- `src/elspeth/engine/adapters.py`
- `tests/engine/test_adapters.py`

**Major changes:**
- `src/elspeth/plugins/protocols.py` - Change SinkProtocol signature
- `src/elspeth/plugins/base.py` - Change BaseSink to match
- `src/elspeth/plugins/sinks/csv_sink.py` - Implement new signature
- `src/elspeth/plugins/sinks/json_sink.py` - Implement new signature
- `src/elspeth/plugins/sinks/database_sink.py` - Implement new signature
- `src/elspeth/engine/executors.py` - Delete *Like protocols
- `src/elspeth/engine/orchestrator.py` - Use protocols directly, fix type alias
- `src/elspeth/cli.py` - Remove SinkAdapter usage

**Test updates:**
- `tests/plugins/sinks/test_csv_sink.py`
- `tests/plugins/sinks/test_json_sink.py`
- `tests/plugins/sinks/test_database_sink.py`
- `tests/engine/test_orchestrator.py`
- `tests/engine/test_integration.py`

---

## Task 1: Update SinkProtocol to be Audit-Aware

**Files:**
- Modify: `src/elspeth/plugins/protocols.py:444-519`
- Test: `tests/plugins/test_protocols.py`

**Step 1.1: Write failing test for new SinkProtocol signature**

```python
# tests/plugins/test_protocols.py - add to TestSinkProtocol class

def test_sink_protocol_write_returns_artifact_descriptor(self) -> None:
    """SinkProtocol.write() must return ArtifactDescriptor for audit."""
    from typing import get_type_hints
    from elspeth.plugins.protocols import SinkProtocol
    from elspeth.contracts.results import ArtifactDescriptor

    hints = get_type_hints(SinkProtocol.write)
    # Should accept list of rows (batch), not single row
    # Should return ArtifactDescriptor, not None
    assert hints["return"] == ArtifactDescriptor
    assert "rows" in hints or "list" in str(hints.get("row", ""))
```

**Step 1.2: Run test to verify it fails**

Run: `pytest tests/plugins/test_protocols.py::TestSinkProtocol::test_sink_protocol_write_returns_artifact_descriptor -v`
Expected: FAIL - current signature returns None

**Step 1.3: Update SinkProtocol signature**

```python
# src/elspeth/plugins/protocols.py - replace lines 480-491

    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: "PluginContext",
    ) -> "ArtifactDescriptor":
        """Write rows to the sink and return artifact info for audit.

        Sinks MUST return ArtifactDescriptor with content_hash and size_bytes.
        This is non-negotiable for audit integrity - the audit trail must
        record what was written, not trust the sink's word for it.

        Args:
            rows: Batch of row data to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with artifact_type, path_or_uri, content_hash, size_bytes
        """
        ...
```

**Step 1.4: Add ArtifactDescriptor import to protocols.py**

```python
# src/elspeth/plugins/protocols.py - add to imports (near top)
from elspeth.contracts.results import ArtifactDescriptor
```

**Step 1.5: Update the docstring example in SinkProtocol**

```python
# src/elspeth/plugins/protocols.py - update Example block in SinkProtocol docstring

    """Protocol for sink plugins.

    Sinks output data to external destinations with full audit support.
    Every write returns an ArtifactDescriptor proving what was written.

    Example:
        class CSVSink:
            name = "csv"
            input_schema = RowSchema
            idempotent = False

            def write(self, rows: list[dict], ctx: PluginContext) -> ArtifactDescriptor:
                for row in rows:
                    self._writer.writerow(row)
                self._file.flush()
                return ArtifactDescriptor.for_file(
                    path=self._path,
                    content_hash=self._compute_hash(),
                    size_bytes=os.path.getsize(self._path),
                )

            def close(self) -> None:
                self._file.close()
    """
```

**Step 1.6: Run test to verify it passes**

Run: `pytest tests/plugins/test_protocols.py::TestSinkProtocol::test_sink_protocol_write_returns_artifact_descriptor -v`
Expected: PASS

**Step 1.7: Commit**

```bash
git add src/elspeth/plugins/protocols.py tests/plugins/test_protocols.py
git commit -m "$(cat <<'EOF'
feat(protocols): make SinkProtocol audit-aware

SinkProtocol.write() now:
- Accepts list[dict] (batch) instead of single dict
- Returns ArtifactDescriptor instead of None

This makes audit a first-class concern in the plugin contract.
Sinks must prove what they wrote.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Update BaseSink Abstract Base Class

**Files:**
- Modify: `src/elspeth/plugins/base.py:237-288`
- Test: `tests/plugins/test_base.py`

**Step 2.1: Write failing test for new BaseSink signature**

```python
# tests/plugins/test_base.py - add new test

def test_base_sink_write_signature_matches_protocol(self) -> None:
    """BaseSink.write() must match SinkProtocol signature."""
    from typing import get_type_hints
    from elspeth.plugins.base import BaseSink
    from elspeth.contracts.results import ArtifactDescriptor

    hints = get_type_hints(BaseSink.write)
    assert hints["return"] == ArtifactDescriptor
```

**Step 2.2: Run test to verify it fails**

Run: `pytest tests/plugins/test_base.py::test_base_sink_write_signature_matches_protocol -v`
Expected: FAIL

**Step 2.3: Update BaseSink.write() signature**

```python
# src/elspeth/plugins/base.py - replace lines 271-278

    @abstractmethod
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> "ArtifactDescriptor":
        """Write rows to the sink and return artifact info.

        Args:
            rows: Batch of rows to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor proving what was written
        """
        ...
```

**Step 2.4: Add ArtifactDescriptor import**

```python
# src/elspeth/plugins/base.py - add to TYPE_CHECKING block
from elspeth.contracts.results import ArtifactDescriptor
```

**Step 2.5: Update BaseSink docstring example**

```python
# src/elspeth/plugins/base.py - update docstring

class BaseSink(ABC):
    """Base class for sink plugins.

    Subclass and implement write(), flush(), close().

    Example:
        class CSVSink(BaseSink):
            name = "csv"
            input_schema = RowSchema
            idempotent = False

            def write(self, rows: list[dict], ctx) -> ArtifactDescriptor:
                for row in rows:
                    self._writer.writerow(row)
                self._file.flush()
                return ArtifactDescriptor.for_file(
                    path=self._path,
                    content_hash=self._compute_hash(),
                    size_bytes=os.path.getsize(self._path),
                )

            def flush(self) -> None:
                self._file.flush()

            def close(self) -> None:
                self._file.close()
    """
```

**Step 2.6: Run test to verify it passes**

Run: `pytest tests/plugins/test_base.py::test_base_sink_write_signature_matches_protocol -v`
Expected: PASS

**Step 2.7: Commit**

```bash
git add src/elspeth/plugins/base.py tests/plugins/test_base.py
git commit -m "$(cat <<'EOF'
feat(base): update BaseSink to match audit-aware SinkProtocol

BaseSink.write() now:
- Accepts list[dict] (batch)
- Returns ArtifactDescriptor

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update CSVSink Implementation

**Files:**
- Modify: `src/elspeth/plugins/sinks/csv_sink.py`
- Test: `tests/plugins/sinks/test_csv_sink.py`

**Step 3.1: Write failing test for new CSVSink**

```python
# tests/plugins/sinks/test_csv_sink.py - add new test

def test_csv_sink_write_returns_artifact_descriptor(tmp_path: Path) -> None:
    """CSVSink.write() returns ArtifactDescriptor with hash and size."""
    from elspeth.plugins.sinks.csv_sink import CSVSink
    from elspeth.plugins.context import PluginContext
    from elspeth.contracts.results import ArtifactDescriptor

    output_path = tmp_path / "output.csv"
    sink = CSVSink({"path": str(output_path)})
    ctx = PluginContext(run_id="test", config={})

    rows = [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]
    result = sink.write(rows, ctx)

    assert isinstance(result, ArtifactDescriptor)
    assert result.artifact_type == "file"
    assert result.content_hash != ""
    assert result.size_bytes > 0
    assert output_path.exists()

    sink.close()
```

**Step 3.2: Run test to verify it fails**

Run: `pytest tests/plugins/sinks/test_csv_sink.py::test_csv_sink_write_returns_artifact_descriptor -v`
Expected: FAIL - returns None

**Step 3.3: Implement new CSVSink.write()**

```python
# src/elspeth/plugins/sinks/csv_sink.py - complete rewrite

"""CSV sink plugin for ELSPETH.

Writes rows to CSV files with audit-required artifact metadata.
"""

import csv
import hashlib
import os
from collections.abc import Sequence
from typing import IO, Any

from elspeth.contracts import PluginSchema
from elspeth.contracts.results import ArtifactDescriptor
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

    Config options:
        path: Path to output CSV file (required)
        delimiter: Field delimiter (default: ",")
        encoding: File encoding (default: "utf-8")
    """

    name = "csv"
    input_schema = CSVInputSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = CSVSinkConfig.from_dict(config)
        self._path = cfg.resolved_path()
        self._delimiter = cfg.delimiter
        self._encoding = cfg.encoding

        self._file: IO[str] | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: Sequence[str] | None = None

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
        """Write rows to CSV and return artifact descriptor.

        Creates file and header on first row. Returns hash of final file.
        """
        for row in rows:
            self._write_row(row)

        self.flush()

        return ArtifactDescriptor.for_file(
            path=self._path,
            content_hash=self._compute_file_hash(),
            size_bytes=os.path.getsize(self._path) if os.path.exists(self._path) else 0,
        )

    def _write_row(self, row: dict[str, Any]) -> None:
        """Write a single row, initializing file if needed."""
        if self._file is None:
            self._fieldnames = list(row.keys())
            self._file = open(self._path, "w", encoding=self._encoding, newline="")  # noqa: SIM115
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._fieldnames,
                delimiter=self._delimiter,
            )
            self._writer.writeheader()

        self._writer.writerow(row)  # type: ignore[union-attr]

    def _compute_file_hash(self) -> str:
        """Compute SHA-256 hash of file contents."""
        if not os.path.exists(self._path):
            return ""
        sha256 = hashlib.sha256()
        with open(self._path, "rb") as f:
            while chunk := f.read(65536):
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
```

**Step 3.4: Run test to verify it passes**

Run: `pytest tests/plugins/sinks/test_csv_sink.py::test_csv_sink_write_returns_artifact_descriptor -v`
Expected: PASS

**Step 3.5: Run all CSV sink tests**

Run: `pytest tests/plugins/sinks/test_csv_sink.py -v`
Expected: Some may fail due to old signature - update them

**Step 3.6: Update existing CSV sink tests to use new signature**

Any test calling `sink.write(row, ctx)` (single row) must change to `sink.write([row], ctx)` (list).

**Step 3.7: Run all CSV sink tests again**

Run: `pytest tests/plugins/sinks/test_csv_sink.py -v`
Expected: PASS

**Step 3.8: Commit**

```bash
git add src/elspeth/plugins/sinks/csv_sink.py tests/plugins/sinks/test_csv_sink.py
git commit -m "$(cat <<'EOF'
feat(csv-sink): implement audit-aware write()

CSVSink.write() now:
- Accepts batch of rows
- Returns ArtifactDescriptor with file hash and size
- Computes SHA-256 of written file for audit integrity

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update JSONSink Implementation

**Files:**
- Modify: `src/elspeth/plugins/sinks/json_sink.py`
- Test: `tests/plugins/sinks/test_json_sink.py`

**Step 4.1: Write failing test**

```python
# tests/plugins/sinks/test_json_sink.py - add test

def test_json_sink_write_returns_artifact_descriptor(tmp_path: Path) -> None:
    """JSONSink.write() returns ArtifactDescriptor."""
    from elspeth.plugins.sinks.json_sink import JSONSink
    from elspeth.plugins.context import PluginContext
    from elspeth.contracts.results import ArtifactDescriptor

    output_path = tmp_path / "output.json"
    sink = JSONSink({"path": str(output_path)})
    ctx = PluginContext(run_id="test", config={})

    rows = [{"id": 1}, {"id": 2}]
    result = sink.write(rows, ctx)

    assert isinstance(result, ArtifactDescriptor)
    assert result.artifact_type == "file"
    assert result.content_hash != ""
    assert result.size_bytes > 0

    sink.close()
```

**Step 4.2: Run test to verify it fails**

Run: `pytest tests/plugins/sinks/test_json_sink.py::test_json_sink_write_returns_artifact_descriptor -v`
Expected: FAIL

**Step 4.3: Update JSONSink implementation**

Follow same pattern as CSVSink - batch write, return ArtifactDescriptor with hash/size.

**Step 4.4: Run tests**

Run: `pytest tests/plugins/sinks/test_json_sink.py -v`
Expected: PASS (after updating old tests)

**Step 4.5: Commit**

```bash
git add src/elspeth/plugins/sinks/json_sink.py tests/plugins/sinks/test_json_sink.py
git commit -m "$(cat <<'EOF'
feat(json-sink): implement audit-aware write()

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update DatabaseSink Implementation

**Files:**
- Modify: `src/elspeth/plugins/sinks/database_sink.py`
- Test: `tests/plugins/sinks/test_database_sink.py`

**Step 5.1: Write failing test**

```python
# tests/plugins/sinks/test_database_sink.py - add test

def test_database_sink_write_returns_artifact_descriptor(tmp_path: Path) -> None:
    """DatabaseSink.write() returns ArtifactDescriptor."""
    from elspeth.plugins.sinks.database_sink import DatabaseSink
    from elspeth.plugins.context import PluginContext
    from elspeth.contracts.results import ArtifactDescriptor

    db_path = tmp_path / "test.db"
    sink = DatabaseSink({"url": f"sqlite:///{db_path}", "table": "results"})
    ctx = PluginContext(run_id="test", config={})

    rows = [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]
    result = sink.write(rows, ctx)

    assert isinstance(result, ArtifactDescriptor)
    assert result.artifact_type == "database"
    assert result.content_hash != ""  # Hash of payload before insert
    assert result.size_bytes > 0

    sink.close()
```

**Step 5.2: Run test to verify it fails**

Run: `pytest tests/plugins/sinks/test_database_sink.py::test_database_sink_write_returns_artifact_descriptor -v`
Expected: FAIL

**Step 5.3: Update DatabaseSink implementation**

For database sinks, hash the canonical JSON of rows BEFORE insert (this proves what was sent).

**Step 5.4: Run tests**

Run: `pytest tests/plugins/sinks/test_database_sink.py -v`
Expected: PASS

**Step 5.5: Commit**

```bash
git add src/elspeth/plugins/sinks/database_sink.py tests/plugins/sinks/test_database_sink.py
git commit -m "$(cat <<'EOF'
feat(database-sink): implement audit-aware write()

DatabaseSink.write() returns ArtifactDescriptor with:
- Hash of canonical JSON payload (proves what was sent)
- Payload size in bytes
- Row count in metadata

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Delete SinkAdapter and SinkLike

**Files:**
- Delete: `src/elspeth/engine/adapters.py`
- Delete: `tests/engine/test_adapters.py`
- Modify: `src/elspeth/engine/__init__.py` (remove SinkAdapter export)
- Modify: `src/elspeth/engine/executors.py` (delete SinkLike protocol)

**Step 6.1: Remove SinkAdapter from engine/__init__.py exports**

```python
# src/elspeth/engine/__init__.py - remove these lines
from elspeth.engine.adapters import SinkAdapter
# and from __all__:
"SinkAdapter",
```

**Step 6.2: Delete SinkLike from executors.py**

Remove lines 668-692 (the SinkLike Protocol class).

**Step 6.3: Update SinkExecutor to use SinkProtocol**

```python
# src/elspeth/engine/executors.py - update import and type hint
from elspeth.plugins.protocols import SinkProtocol

# In SinkExecutor.write():
def write(
    self,
    sink: SinkProtocol,  # Changed from SinkLike
    ...
```

**Step 6.4: Delete adapters.py file**

```bash
rm src/elspeth/engine/adapters.py
```

**Step 6.5: Delete test_adapters.py file**

```bash
rm tests/engine/test_adapters.py
```

**Step 6.6: Run engine tests**

Run: `pytest tests/engine/ -v`
Expected: Some failures in orchestrator tests - fix in next task

**Step 6.7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(engine): delete SinkAdapter and SinkLike

SinkProtocol is now the single source of truth for sink interfaces.
No adapter needed - sinks return ArtifactDescriptor directly.

BREAKING: SinkAdapter removed from public API

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update CLI to Use Sinks Directly

**Files:**
- Modify: `src/elspeth/cli.py:260-292`

**Step 7.1: Remove SinkAdapter usage from CLI**

```python
# src/elspeth/cli.py - replace sink creation block

    # Build sinks directly (no adapter needed - SinkProtocol is audit-aware)
    sinks: dict[str, SinkProtocol] = {}
    for sink_name, sink_config in config.sinks.items():
        sink_plugin = sink_config.plugin
        sink_options = dict(sink_config.options)

        if sink_plugin == "csv":
            sinks[sink_name] = CSVSink(sink_options)
        elif sink_plugin == "json":
            sinks[sink_name] = JSONSink(sink_options)
        elif sink_plugin == "database":
            sinks[sink_name] = DatabaseSink(sink_options)
        else:
            raise ValueError(f"Unknown sink plugin: {sink_plugin}")
```

**Step 7.2: Remove SinkAdapter import**

```python
# src/elspeth/cli.py - remove this import
from elspeth.engine.adapters import SinkAdapter
```

**Step 7.3: Update PipelineConfig type hints (remove type: ignore)**

The `sinks` dict is now `dict[str, SinkProtocol]` which matches what orchestrator expects.

**Step 7.4: Run CLI tests**

Run: `pytest tests/cli/ -v` (if exists) or test manually

**Step 7.5: Commit**

```bash
git add src/elspeth/cli.py
git commit -m "$(cat <<'EOF'
refactor(cli): use sinks directly without adapter

SinkProtocol is now audit-aware, so no adapter wrapper needed.
Removes type: ignore comments - types now align.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update Orchestrator Types

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py:21-50`

**Step 8.1: Fix TransformLike name collision**

```python
# src/elspeth/engine/orchestrator.py - rename the union type

# Change this:
TransformLike = BaseTransform | BaseGate | BaseAggregation

# To this:
RowPlugin = BaseTransform | BaseGate | BaseAggregation
```

**Step 8.2: Update PipelineConfig to use SinkProtocol**

```python
# src/elspeth/engine/orchestrator.py

from elspeth.plugins.protocols import SinkProtocol, SourceProtocol

@dataclass
class PipelineConfig:
    source: SourceProtocol
    transforms: list[RowPlugin]
    sinks: dict[str, SinkProtocol]  # Now uses SinkProtocol directly
    config: dict[str, Any] = field(default_factory=dict)
```

**Step 8.3: Remove SinkLike import**

```python
# src/elspeth/engine/orchestrator.py - remove this line
from elspeth.engine.executors import SinkLike
```

**Step 8.4: Remove type: ignore comments in _execute_run**

The `sink_executor.write(sink=sink, ...)` call should now type-check correctly.

**Step 8.5: Run orchestrator tests**

Run: `pytest tests/engine/test_orchestrator.py -v`
Expected: PASS

**Step 8.6: Commit**

```bash
git add src/elspeth/engine/orchestrator.py
git commit -m "$(cat <<'EOF'
refactor(orchestrator): use SinkProtocol directly

- Rename TransformLike -> RowPlugin (fixes name collision)
- PipelineConfig.sinks now typed as dict[str, SinkProtocol]
- Remove type: ignore hacks - types align properly

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Delete Unnecessary *Like Protocols

**Files:**
- Modify: `src/elspeth/engine/executors.py`

**Step 9.1: Delete TransformLike, GateLike, AggregationLike**

These protocols (lines 75-83, 212-220, 444-465) are identical to the full protocols and add no value.

**Step 9.2: Update executor type hints to use full protocols**

```python
# src/elspeth/engine/executors.py

from elspeth.plugins.protocols import (
    TransformProtocol,
    GateProtocol,
    AggregationProtocol,
    SinkProtocol,
)

class TransformExecutor:
    def execute_transform(
        self,
        transform: TransformProtocol,  # Was TransformLike
        ...
    )

class GateExecutor:
    def execute_gate(
        self,
        gate: GateProtocol,  # Was GateLike
        ...
    )

class AggregationExecutor:
    def accept(
        self,
        aggregation: AggregationProtocol,  # Was AggregationLike
        ...
    )
```

**Step 9.3: Run all executor tests**

Run: `pytest tests/engine/test_executors.py -v` (if exists)

**Step 9.4: Run full test suite**

Run: `pytest tests/ -v`
Expected: PASS

**Step 9.5: Commit**

```bash
git add src/elspeth/engine/executors.py
git commit -m "$(cat <<'EOF'
refactor(executors): delete redundant *Like protocols

TransformLike, GateLike, AggregationLike were identical to the full
protocols. Use full protocols directly for clarity.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final Integration Test

**Files:**
- Test: `tests/engine/test_integration.py`

**Step 10.1: Run full integration test suite**

Run: `pytest tests/engine/test_integration.py -v`
Expected: PASS

**Step 10.2: Run full test suite with coverage**

Run: `pytest tests/ --cov=src/elspeth --cov-report=term-missing`
Expected: PASS, no regressions

**Step 10.3: Run type checker**

Run: `mypy src/elspeth/`
Expected: PASS (fewer type: ignore comments than before)

**Step 10.4: Run linter**

Run: `ruff check src/elspeth/`
Expected: PASS

**Step 10.5: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test: verify interface unification

All tests pass. Type checker happy. Linter happy.

Summary of changes:
- SinkProtocol is now audit-aware (batch + ArtifactDescriptor)
- SinkAdapter deleted (no longer needed)
- *Like protocols deleted (redundant)
- TransformLike renamed to RowPlugin (fixes collision)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Before | After |
|--------|-------|
| `SinkProtocol.write(row) -> None` | `SinkProtocol.write(rows) -> ArtifactDescriptor` |
| `SinkLike` in executors.py | Deleted |
| `SinkAdapter` in adapters.py | Deleted |
| `TransformLike` (protocol) | Deleted |
| `GateLike` (protocol) | Deleted |
| `AggregationLike` (protocol) | Deleted |
| `TransformLike` (union alias) | Renamed to `RowPlugin` |

**Lines of code removed:** ~400 (adapters.py, test_adapters.py, *Like protocols)

**Type safety improved:** Removed 4 `type: ignore` comments from cli.py and orchestrator.py
