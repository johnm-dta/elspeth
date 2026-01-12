# Phase 4: CLI & Basic I/O (Tasks 1-14)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the user-facing CLI with Typer, implement basic source/sink plugins (CSV, JSON, Database), add the `explain` command with Textual TUI, and integrate structlog for structured logging.

**Architecture:** The CLI uses Typer for command parsing with subcommands (`run`, `explain`, `validate`, `plugins`). Source/sink plugins follow the Phase 2 protocol patterns with schema validation. The `explain` command composes LandscapeRecorder query methods into a lineage traversal, rendered via Textual widgets. structlog provides structured logging that complements OpenTelemetry spans.

**Tech Stack:** Python 3.11+, Typer (CLI), Textual (TUI), structlog (logging), pandas (CSV/data handling)

**Dependencies:**
- Phase 1: `elspeth.core.canonical`, `elspeth.core.config`, `elspeth.core.dag`, `elspeth.core.payload_store`
- Phase 2: `elspeth.plugins` (protocols, base, manager, schemas, results, context)
- Phase 3: `elspeth.core.landscape` (LandscapeDB, LandscapeRecorder), `elspeth.engine` (Orchestrator, PipelineConfig, RunResult)

---

## ⚠️ SINK INTERFACE NOTE

**Phase 2 `SinkProtocol` vs Phase 3B `SinkLike`:**

| Layer | Signature | Returns |
|-------|-----------|---------|
| Phase 2 `SinkProtocol` | `write(row: dict, ctx) -> None` | Nothing |
| Phase 3B `SinkLike` | `write(rows: list[dict], ctx) -> dict` | Artifact info |

**IMPORTANT:** Phase 3B `SinkExecutor` expects the bulk `SinkLike` interface - it does NOT loop over rows internally. It calls `sink.write(rows, ctx)` once with the full list.

Phase 4 sinks implement the Phase 2 row-wise protocol (simpler for plugin authors). **Phase 4 must provide a `SinkAdapter`** that:
1. Implements `SinkLike` (bulk interface expected by SinkExecutor)
2. Wraps a Phase 2 sink
3. Loops calling `sink.write(row, ctx)` for each row
4. Calls `sink.flush()` after writing
5. Generates artifact info based on artifact descriptor kind

**Lifecycle:** Phase 3B's Orchestrator calls `sink.close()` for all sinks in its finally block. The adapter's `close()` method delegates to the wrapped sink.

**This means:** Phase 4 sinks are correct for plugin authors. The CLI `run` command wraps them in `SinkAdapter` before passing to Orchestrator.

---

## Task 1: CLI Foundation - Typer App Scaffold

**Context:** Create the main CLI entry point using Typer. The entry point `elspeth = "elspeth.cli:app"` is already defined in pyproject.toml but the module doesn't exist.

**Files:**
- Create: `src/elspeth/cli.py`
- Create: `tests/cli/test_cli.py`
- Create: `tests/cli/__init__.py`

### Step 1: Write the failing test

```python
# tests/cli/__init__.py
"""CLI tests."""

# tests/cli/test_cli.py
"""Tests for ELSPETH CLI."""

import pytest
from typer.testing import CliRunner

# IMPORTANT: mix_stderr=True ensures error messages are captured in stdout
# for consistent test assertions (Typer/Click writes errors to stderr by default)
runner = CliRunner(mix_stderr=True)


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_exists(self) -> None:
        """CLI app can be imported."""
        from elspeth.cli import app

        assert app is not None

    def test_version_flag(self) -> None:
        """--version shows version info."""
        from elspeth.cli import app

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "elspeth" in result.stdout.lower()

    def test_help_flag(self) -> None:
        """--help shows available commands."""
        from elspeth.cli import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "explain" in result.stdout
        assert "validate" in result.stdout
        assert "plugins" in result.stdout
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.cli'`

### Step 3: Write minimal implementation

```python
# src/elspeth/cli.py
"""ELSPETH Command Line Interface.

Entry point for the elspeth CLI tool.
"""

from typing import Optional

import typer

from elspeth import __version__

app = typer.Typer(
    name="elspeth",
    help="ELSPETH: Auditable Sense/Decide/Act pipelines.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"elspeth version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """ELSPETH: Auditable Sense/Decide/Act pipelines."""
    pass


# === Subcommand stubs (to be implemented in later tasks) ===


@app.command()
def run(
    settings: str = typer.Option(
        ...,
        "--settings",
        "-s",
        help="Path to settings YAML file.",
    ),
) -> None:
    """Execute a pipeline run."""
    typer.echo(f"Run command not yet implemented. Settings: {settings}")
    raise typer.Exit(1)


@app.command()
def explain(
    run_id: str = typer.Option(
        ...,
        "--run",
        "-r",
        help="Run ID to explain (or 'latest').",
    ),
    row: Optional[str] = typer.Option(
        None,
        "--row",
        help="Row ID or index to explain.",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        help="Token ID for precise lineage.",
    ),
) -> None:
    """Explain lineage for a row or token."""
    typer.echo(f"Explain command not yet implemented. Run: {run_id}")
    raise typer.Exit(1)


@app.command()
def validate(
    settings: str = typer.Option(
        ...,
        "--settings",
        "-s",
        help="Path to settings YAML file.",
    ),
) -> None:
    """Validate pipeline configuration without running."""
    typer.echo(f"Validate command not yet implemented. Settings: {settings}")
    raise typer.Exit(1)


# Plugins subcommand group
plugins_app = typer.Typer(help="Plugin management commands.")
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list() -> None:
    """List available plugins."""
    typer.echo("Plugins list not yet implemented.")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_cli.py -v`
Expected: PASS (3 tests)

### Step 5: Commit

```bash
git add tests/cli/ src/elspeth/cli.py
git commit -m "feat(cli): add Typer CLI scaffold with subcommand stubs"
```

---

## Task 2: CSV Source Plugin

**Context:** Implement a CSV source that reads rows from a CSV file. Uses pandas for robust CSV parsing. Follows the SourceProtocol from Phase 2.

**Files:**
- Create: `src/elspeth/plugins/sources/csv_source.py`
- Create: `src/elspeth/plugins/sources/__init__.py` (replace placeholder)
- Create: `tests/plugins/sources/__init__.py`
- Create: `tests/plugins/sources/test_csv_source.py`

### Step 1: Write the failing test

```python
# tests/plugins/sources/__init__.py
"""Source plugin tests."""

# tests/plugins/sources/test_csv_source.py
"""Tests for CSV source plugin."""

from pathlib import Path
from typing import Iterator

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SourceProtocol


class TestCSVSource:
    """Tests for CSVSource plugin."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create a sample CSV file."""
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text("id,name,value\n1,alice,100\n2,bob,200\n3,carol,300\n")
        return csv_file

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """CSVSource implements SourceProtocol."""
        from elspeth.plugins.sources.csv_source import CSVSource

        assert isinstance(CSVSource, type)
        # Runtime check via Protocol
        source = CSVSource({"path": "/tmp/test.csv"})
        assert isinstance(source, SourceProtocol)

    def test_has_required_attributes(self) -> None:
        """CSVSource has name and output_schema."""
        from elspeth.plugins.sources.csv_source import CSVSource

        assert CSVSource.name == "csv"
        assert hasattr(CSVSource, "output_schema")

    def test_load_yields_rows(self, sample_csv: Path, ctx: PluginContext) -> None:
        """load() yields dict rows from CSV."""
        from elspeth.plugins.sources.csv_source import CSVSource

        source = CSVSource({"path": str(sample_csv)})
        rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[0] == {"id": "1", "name": "alice", "value": "100"}
        assert rows[1]["name"] == "bob"
        assert rows[2]["value"] == "300"

    def test_load_with_delimiter(self, tmp_path: Path, ctx: PluginContext) -> None:
        """CSV with custom delimiter."""
        from elspeth.plugins.sources.csv_source import CSVSource

        csv_file = tmp_path / "semicolon.csv"
        csv_file.write_text("id;name;value\n1;alice;100\n")

        source = CSVSource({"path": str(csv_file), "delimiter": ";"})
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert rows[0]["name"] == "alice"

    def test_load_with_encoding(self, tmp_path: Path, ctx: PluginContext) -> None:
        """CSV with non-UTF8 encoding."""
        from elspeth.plugins.sources.csv_source import CSVSource

        csv_file = tmp_path / "latin1.csv"
        csv_file.write_bytes(b"id,name\n1,caf\xe9\n")

        source = CSVSource({"path": str(csv_file), "encoding": "latin-1"})
        rows = list(source.load(ctx))

        assert rows[0]["name"] == "caf\xe9"

    def test_close_is_idempotent(self, sample_csv: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sources.csv_source import CSVSource

        source = CSVSource({"path": str(sample_csv)})
        list(source.load(ctx))  # Consume iterator
        source.close()
        source.close()  # Should not raise

    def test_file_not_found_raises(self, ctx: PluginContext) -> None:
        """Missing file raises FileNotFoundError."""
        from elspeth.plugins.sources.csv_source import CSVSource

        source = CSVSource({"path": "/nonexistent/file.csv"})
        with pytest.raises(FileNotFoundError):
            list(source.load(ctx))
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/sources/test_csv_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.plugins.sources.csv_source'`

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/sources/csv_source.py
"""CSV source plugin for ELSPETH.

Loads rows from CSV files using pandas for robust parsing.
"""

from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from elspeth.plugins.base import BaseSource
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schemas import PluginSchema


class CSVOutputSchema(PluginSchema):
    """Dynamic schema - CSV columns are determined at runtime."""

    model_config = {"extra": "allow"}


class CSVSource(BaseSource):
    """Load rows from a CSV file.

    Config options:
        path: Path to CSV file (required)
        delimiter: Field delimiter (default: ",")
        encoding: File encoding (default: "utf-8")
        skip_rows: Number of header rows to skip (default: 0)
    """

    name = "csv"
    output_schema = CSVOutputSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._path = Path(config["path"])
        self._delimiter = config.get("delimiter", ",")
        self._encoding = config.get("encoding", "utf-8")
        self._skip_rows = config.get("skip_rows", 0)
        self._dataframe: pd.DataFrame | None = None

    def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Load rows from CSV file.

        Yields:
            Dict for each row with column names as keys.

        Raises:
            FileNotFoundError: If CSV file does not exist.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"CSV file not found: {self._path}")

        self._dataframe = pd.read_csv(
            self._path,
            delimiter=self._delimiter,
            encoding=self._encoding,
            skiprows=self._skip_rows,
            dtype=str,  # Keep all values as strings for consistent handling
            keep_default_na=False,  # Don't convert empty strings to NaN
        )

        for _, row in self._dataframe.iterrows():
            yield row.to_dict()

    def close(self) -> None:
        """Release resources."""
        self._dataframe = None
```

```python
# src/elspeth/plugins/sources/__init__.py
"""Built-in source plugins for ELSPETH.

Sources load data into the pipeline. Exactly one source per run.
"""

from elspeth.plugins.sources.csv_source import CSVSource

__all__ = ["CSVSource"]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/sources/test_csv_source.py -v`
Expected: PASS (7 tests)

### Step 5: Commit

```bash
git add src/elspeth/plugins/sources/ tests/plugins/sources/
git commit -m "feat(plugins): add CSVSource plugin"
```

---

## Task 3: JSON Source Plugin

**Context:** Implement a JSON source that reads rows from JSON files. Supports both JSON array format and JSONL (JSON Lines) format.

**Files:**
- Create: `src/elspeth/plugins/sources/json_source.py`
- Modify: `src/elspeth/plugins/sources/__init__.py`
- Create: `tests/plugins/sources/test_json_source.py`

### Step 1: Write the failing test

```python
# tests/plugins/sources/test_json_source.py
"""Tests for JSON source plugin."""

import json
from pathlib import Path

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SourceProtocol


class TestJSONSource:
    """Tests for JSONSource plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """JSONSource implements SourceProtocol."""
        from elspeth.plugins.sources.json_source import JSONSource

        source = JSONSource({"path": "/tmp/test.json"})
        assert isinstance(source, SourceProtocol)

    def test_has_required_attributes(self) -> None:
        """JSONSource has name and output_schema."""
        from elspeth.plugins.sources.json_source import JSONSource

        assert JSONSource.name == "json"
        assert hasattr(JSONSource, "output_schema")

    def test_load_json_array(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Load rows from JSON array file."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        data = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
        ]
        json_file.write_text(json.dumps(data))

        source = JSONSource({"path": str(json_file)})
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0] == {"id": 1, "name": "alice"}
        assert rows[1]["name"] == "bob"

    def test_load_jsonl(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Load rows from JSONL file."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            '{"id": 1, "name": "alice"}\n'
            '{"id": 2, "name": "bob"}\n'
            '{"id": 3, "name": "carol"}\n'
        )

        source = JSONSource({"path": str(jsonl_file), "format": "jsonl"})
        rows = list(source.load(ctx))

        assert len(rows) == 3
        assert rows[2]["name"] == "carol"

    def test_auto_detect_jsonl_by_extension(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Auto-detect JSONL format from .jsonl extension."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"id": 1}\n{"id": 2}\n')

        source = JSONSource({"path": str(jsonl_file)})  # No format specified
        rows = list(source.load(ctx))

        assert len(rows) == 2

    def test_json_object_with_data_key(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Load rows from nested JSON object using data_key."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "wrapped.json"
        data = {
            "metadata": {"count": 2},
            "results": [{"id": 1}, {"id": 2}],
        }
        json_file.write_text(json.dumps(data))

        source = JSONSource({"path": str(json_file), "data_key": "results"})
        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert rows[0] == {"id": 1}

    def test_empty_lines_ignored_in_jsonl(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Empty lines in JSONL are ignored."""
        from elspeth.plugins.sources.json_source import JSONSource

        jsonl_file = tmp_path / "sparse.jsonl"
        jsonl_file.write_text('{"id": 1}\n\n{"id": 2}\n\n')

        source = JSONSource({"path": str(jsonl_file), "format": "jsonl"})
        rows = list(source.load(ctx))

        assert len(rows) == 2

    def test_file_not_found_raises(self, ctx: PluginContext) -> None:
        """Missing file raises FileNotFoundError."""
        from elspeth.plugins.sources.json_source import JSONSource

        source = JSONSource({"path": "/nonexistent/file.json"})
        with pytest.raises(FileNotFoundError):
            list(source.load(ctx))

    def test_close_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """close() can be called multiple times."""
        from elspeth.plugins.sources.json_source import JSONSource

        json_file = tmp_path / "data.json"
        json_file.write_text("[]")

        source = JSONSource({"path": str(json_file)})
        list(source.load(ctx))
        source.close()
        source.close()  # Should not raise
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/sources/test_json_source.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/sources/json_source.py
"""JSON source plugin for ELSPETH.

Loads rows from JSON files. Supports JSON array and JSONL formats.
"""

import json
from pathlib import Path
from typing import Any, Iterator

from elspeth.plugins.base import BaseSource
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schemas import PluginSchema


class JSONOutputSchema(PluginSchema):
    """Dynamic schema - JSON fields determined at runtime."""

    model_config = {"extra": "allow"}


class JSONSource(BaseSource):
    """Load rows from a JSON file.

    Config options:
        path: Path to JSON file (required)
        format: "json" (array) or "jsonl" (lines). Auto-detected from extension if not set.
        data_key: Key to extract array from JSON object (e.g., "results")
        encoding: File encoding (default: "utf-8")
    """

    name = "json"
    output_schema = JSONOutputSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._path = Path(config["path"])
        self._encoding = config.get("encoding", "utf-8")
        self._data_key = config.get("data_key")

        # Auto-detect format from extension if not specified
        fmt = config.get("format")
        if fmt is None:
            fmt = "jsonl" if self._path.suffix == ".jsonl" else "json"
        self._format = fmt

    def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Load rows from JSON file.

        Yields:
            Dict for each row.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If JSON is invalid or not an array.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"JSON file not found: {self._path}")

        if self._format == "jsonl":
            yield from self._load_jsonl()
        else:
            yield from self._load_json_array()

    def _load_jsonl(self) -> Iterator[dict[str, Any]]:
        """Load from JSONL format (one JSON object per line)."""
        with open(self._path, encoding=self._encoding) as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    yield json.loads(line)

    def _load_json_array(self) -> Iterator[dict[str, Any]]:
        """Load from JSON array format."""
        with open(self._path, encoding=self._encoding) as f:
            data = json.load(f)

        # Extract from nested key if specified
        if self._data_key:
            data = data[self._data_key]

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        yield from data

    def close(self) -> None:
        """Release resources (no-op for JSON source)."""
        pass
```

Update the sources `__init__.py`:

```python
# src/elspeth/plugins/sources/__init__.py
"""Built-in source plugins for ELSPETH.

Sources load data into the pipeline. Exactly one source per run.
"""

from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.plugins.sources.json_source import JSONSource

__all__ = ["CSVSource", "JSONSource"]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/sources/test_json_source.py -v`
Expected: PASS (9 tests)

### Step 5: Commit

```bash
git add src/elspeth/plugins/sources/ tests/plugins/sources/test_json_source.py
git commit -m "feat(plugins): add JSONSource plugin with JSONL support"
```

---

## Task 4: CSV Sink Plugin

**Context:** Implement a CSV sink that writes rows to a CSV file. Follows the SinkProtocol from Phase 2.

**Files:**
- Create: `src/elspeth/plugins/sinks/csv_sink.py`
- Create: `src/elspeth/plugins/sinks/__init__.py` (replace placeholder)
- Create: `tests/plugins/sinks/__init__.py`
- Create: `tests/plugins/sinks/test_csv_sink.py`

### Step 1: Write the failing test

```python
# tests/plugins/sinks/__init__.py
"""Sink plugin tests."""

# tests/plugins/sinks/test_csv_sink.py
"""Tests for CSV sink plugin."""

import csv
from pathlib import Path

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SinkProtocol


class TestCSVSink:
    """Tests for CSVSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """CSVSink implements SinkProtocol."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        sink = CSVSink({"path": "/tmp/test.csv"})
        assert isinstance(sink, SinkProtocol)

    def test_has_required_attributes(self) -> None:
        """CSVSink has name and idempotent flag."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        assert CSVSink.name == "csv"
        assert CSVSink.idempotent is True

    def test_write_creates_file(self, tmp_path: Path, ctx: PluginContext) -> None:
        """write() creates CSV file with header."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        sink.write({"id": 1, "name": "alice"}, ctx)
        sink.write({"id": 2, "name": "bob"}, ctx)
        sink.flush()
        sink.close()

        assert output_file.exists()
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["name"] == "alice"
        assert rows[1]["id"] == "2"

    def test_write_with_custom_delimiter(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Custom delimiter in output."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "delimiter": ";"})

        sink.write({"a": 1, "b": 2}, ctx)
        sink.flush()
        sink.close()

        content = output_file.read_text()
        assert "a;b" in content
        assert "1;2" in content

    def test_columns_inferred_from_first_row(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Column order determined by first row."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        sink.write({"z": 1, "a": 2, "m": 3}, ctx)  # Order: z, a, m
        sink.flush()
        sink.close()

        content = output_file.read_text()
        # Header should reflect first row's order
        first_line = content.split("\n")[0]
        assert first_line == "z,a,m"

    def test_explicit_columns_order(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Explicit columns config controls order."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file), "columns": ["a", "z", "m"]})

        sink.write({"z": 1, "a": 2, "m": 3}, ctx)
        sink.flush()
        sink.close()

        content = output_file.read_text()
        first_line = content.split("\n")[0]
        assert first_line == "a,z,m"

    def test_append_mode(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Append mode adds to existing file."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        output_file.write_text("id,name\n1,alice\n")

        sink = CSVSink({"path": str(output_file), "mode": "append"})
        sink.write({"id": 2, "name": "bob"}, ctx)
        sink.flush()
        sink.close()

        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[1]["name"] == "bob"

    def test_flush_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """flush() can be called multiple times."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        sink.write({"id": 1}, ctx)
        sink.flush()
        sink.flush()  # Should not raise
        sink.close()

    def test_close_without_write(self, tmp_path: Path) -> None:
        """close() works even if no rows written."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})
        sink.close()  # Should not raise
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/sinks/test_csv_sink.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/sinks/csv_sink.py
"""CSV sink plugin for ELSPETH.

Writes rows to CSV files.
"""

import csv
from pathlib import Path
from typing import Any, TextIO

from elspeth.plugins.base import BaseSink
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schemas import PluginSchema


class CSVInputSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""

    model_config = {"extra": "allow"}


class CSVSink(BaseSink):
    """Write rows to a CSV file.

    Config options:
        path: Path to output CSV file (required)
        delimiter: Field delimiter (default: ",")
        mode: "write" (overwrite) or "append" (default: "write")
        columns: Explicit column order (default: infer from first row)
    """

    name = "csv"
    input_schema = CSVInputSchema
    idempotent = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._path = Path(config["path"])
        self._delimiter = config.get("delimiter", ",")
        self._mode = config.get("mode", "write")
        self._columns: list[str] | None = config.get("columns")
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._header_written = False

    def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
        """Write a row to the CSV file.

        On first row, opens file and writes header.
        """
        if self._writer is None:
            self._open_file(row)

        self._writer.writerow(row)

    def _open_file(self, first_row: dict[str, Any]) -> None:
        """Open file and initialize writer with columns from first row."""
        # Determine columns
        if self._columns:
            columns = self._columns
        else:
            columns = list(first_row.keys())

        # Open file
        if self._mode == "append" and self._path.exists():
            self._file = open(self._path, "a", newline="", encoding="utf-8")
            self._header_written = True  # Assume header exists
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, "w", newline="", encoding="utf-8")
            self._header_written = False

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=columns,
            delimiter=self._delimiter,
            extrasaction="ignore",  # Ignore extra fields not in columns
        )

        if not self._header_written:
            self._writer.writeheader()
            self._header_written = True

    def flush(self) -> None:
        """Flush buffered data to disk."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
```

```python
# src/elspeth/plugins/sinks/__init__.py
"""Built-in sink plugins for ELSPETH.

Sinks output data from the pipeline. One or more sinks per run.
"""

from elspeth.plugins.sinks.csv_sink import CSVSink

__all__ = ["CSVSink"]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/sinks/test_csv_sink.py -v`
Expected: PASS (9 tests)

### Step 5: Commit

```bash
git add src/elspeth/plugins/sinks/ tests/plugins/sinks/
git commit -m "feat(plugins): add CSVSink plugin"
```

---

## Task 5: JSON Sink Plugin

**Context:** Implement a JSON sink that writes rows to JSON files. Supports JSON array and JSONL formats.

**Files:**
- Create: `src/elspeth/plugins/sinks/json_sink.py`
- Modify: `src/elspeth/plugins/sinks/__init__.py`
- Create: `tests/plugins/sinks/test_json_sink.py`

### Step 1: Write the failing test

```python
# tests/plugins/sinks/test_json_sink.py
"""Tests for JSON sink plugin."""

import json
from pathlib import Path

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SinkProtocol


class TestJSONSink:
    """Tests for JSONSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """JSONSink implements SinkProtocol."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        sink = JSONSink({"path": "/tmp/test.json"})
        assert isinstance(sink, SinkProtocol)

    def test_has_required_attributes(self) -> None:
        """JSONSink has name and idempotent flag."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        assert JSONSink.name == "json"
        assert JSONSink.idempotent is True

    def test_write_json_array(self, tmp_path: Path, ctx: PluginContext) -> None:
        """Default format writes JSON array."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        sink.write({"id": 1, "name": "alice"}, ctx)
        sink.write({"id": 2, "name": "bob"}, ctx)
        sink.flush()
        sink.close()

        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "alice"

    def test_write_jsonl(self, tmp_path: Path, ctx: PluginContext) -> None:
        """JSONL format writes one object per line."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = JSONSink({"path": str(output_file), "format": "jsonl"})

        sink.write({"id": 1}, ctx)
        sink.write({"id": 2}, ctx)
        sink.flush()
        sink.close()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": 1}
        assert json.loads(lines[1]) == {"id": 2}

    def test_auto_detect_jsonl_by_extension(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """Auto-detect JSONL format from .jsonl extension."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.jsonl"
        sink = JSONSink({"path": str(output_file)})

        sink.write({"id": 1}, ctx)
        sink.flush()
        sink.close()

        # Should be JSONL format
        content = output_file.read_text().strip()
        assert content == '{"id": 1}'  # No array brackets

    def test_pretty_print_option(self, tmp_path: Path, ctx: PluginContext) -> None:
        """pretty=True formats JSON with indentation."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file), "pretty": True})

        sink.write({"id": 1, "name": "alice"}, ctx)
        sink.flush()
        sink.close()

        content = output_file.read_text()
        assert "\n" in content  # Has newlines
        assert "  " in content  # Has indentation

    def test_flush_is_idempotent(self, tmp_path: Path, ctx: PluginContext) -> None:
        """flush() can be called multiple times."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        sink.write({"id": 1}, ctx)
        sink.flush()
        sink.flush()  # Should not raise
        sink.close()

    def test_close_without_write_creates_empty(
        self, tmp_path: Path, ctx: PluginContext
    ) -> None:
        """close() without writes creates empty array (JSON) or empty file (JSONL)."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})
        sink.flush()
        sink.close()

        data = json.loads(output_file.read_text())
        assert data == []
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/sinks/test_json_sink.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/sinks/json_sink.py
"""JSON sink plugin for ELSPETH.

Writes rows to JSON files. Supports JSON array and JSONL formats.
"""

import json
from pathlib import Path
from typing import Any, TextIO

from elspeth.plugins.base import BaseSink
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schemas import PluginSchema


class JSONInputSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""

    model_config = {"extra": "allow"}


class JSONSink(BaseSink):
    """Write rows to a JSON file.

    Config options:
        path: Path to output file (required)
        format: "json" (array) or "jsonl" (lines). Auto-detected from extension.
        pretty: Pretty-print JSON with indentation (default: False)
    """

    name = "json"
    input_schema = JSONInputSchema
    idempotent = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._path = Path(config["path"])
        self._pretty = config.get("pretty", False)

        # Auto-detect format from extension if not specified
        fmt = config.get("format")
        if fmt is None:
            fmt = "jsonl" if self._path.suffix == ".jsonl" else "json"
        self._format = fmt

        self._rows: list[dict[str, Any]] = []
        self._file: TextIO | None = None

    def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
        """Buffer a row for output."""
        if self._format == "jsonl":
            # JSONL: write immediately
            if self._file is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self._path, "w", encoding="utf-8")
            self._file.write(json.dumps(row) + "\n")
        else:
            # JSON array: buffer until flush/close
            self._rows.append(row)

    def flush(self) -> None:
        """Flush buffered data to disk."""
        if self._format == "jsonl":
            if self._file is not None:
                self._file.flush()
        else:
            # JSON array: write complete array
            self._write_json_array()

    def _write_json_array(self) -> None:
        """Write buffered rows as JSON array."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if self._pretty else None
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._rows, f, indent=indent)

    def close(self) -> None:
        """Close the file."""
        if self._format == "jsonl":
            if self._file is not None:
                self._file.close()
                self._file = None
        else:
            # Ensure array is written on close
            self._write_json_array()
```

Update sinks `__init__.py`:

```python
# src/elspeth/plugins/sinks/__init__.py
"""Built-in sink plugins for ELSPETH.

Sinks output data from the pipeline. One or more sinks per run.
"""

from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.json_sink import JSONSink

__all__ = ["CSVSink", "JSONSink"]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/sinks/test_json_sink.py -v`
Expected: PASS (8 tests)

### Step 5: Commit

```bash
git add src/elspeth/plugins/sinks/ tests/plugins/sinks/test_json_sink.py
git commit -m "feat(plugins): add JSONSink plugin with JSONL support"
```

---

## Task 6: Database Sink Plugin

**Context:** Implement a database sink using SQLAlchemy Core. Writes rows to a database table.

**Files:**
- Create: `src/elspeth/plugins/sinks/database_sink.py`
- Modify: `src/elspeth/plugins/sinks/__init__.py`
- Create: `tests/plugins/sinks/test_database_sink.py`

### Step 1: Write the failing test

```python
# tests/plugins/sinks/test_database_sink.py
"""Tests for Database sink plugin."""

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import SinkProtocol


class TestDatabaseSink:
    """Tests for DatabaseSink plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a test database with a table."""
        db_path = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_path}")
        metadata = MetaData()
        Table(
            "results",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String),
            Column("value", Integer),
        )
        metadata.create_all(engine)
        return f"sqlite:///{db_path}"

    def test_implements_protocol(self) -> None:
        """DatabaseSink implements SinkProtocol."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": "sqlite:///:memory:", "table": "test"})
        assert isinstance(sink, SinkProtocol)

    def test_has_required_attributes(self) -> None:
        """DatabaseSink has name and idempotent flag."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        assert DatabaseSink.name == "database"
        # Not idempotent by default (INSERT can duplicate)
        assert DatabaseSink.idempotent is False

    def test_write_inserts_rows(self, test_db: str, ctx: PluginContext) -> None:
        """write() inserts rows into table."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": test_db, "table": "results"})

        sink.write({"name": "alice", "value": 100}, ctx)
        sink.write({"name": "bob", "value": 200}, ctx)
        sink.flush()
        sink.close()

        # Verify data
        engine = create_engine(test_db)
        with engine.connect() as conn:
            result = conn.execute(select(Table("results", MetaData(), autoload_with=engine)))
            rows = result.fetchall()

        assert len(rows) == 2
        assert rows[0].name == "alice"
        assert rows[1].value == 200

    def test_batch_size_controls_commit(self, test_db: str, ctx: PluginContext) -> None:
        """batch_size config controls commit frequency."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": test_db, "table": "results", "batch_size": 2})

        sink.write({"name": "a", "value": 1}, ctx)
        sink.write({"name": "b", "value": 2}, ctx)
        # After 2 rows, should auto-flush
        sink.write({"name": "c", "value": 3}, ctx)
        sink.close()

        engine = create_engine(test_db)
        with engine.connect() as conn:
            result = conn.execute(select(Table("results", MetaData(), autoload_with=engine)))
            rows = result.fetchall()

        assert len(rows) == 3

    def test_auto_create_table(self, tmp_path, ctx: PluginContext) -> None:
        """auto_create=True creates table from first row."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        db_path = tmp_path / "new.db"
        sink = DatabaseSink({
            "url": f"sqlite:///{db_path}",
            "table": "dynamic_table",
            "auto_create": True,
        })

        sink.write({"name": "alice", "score": 95.5}, ctx)
        sink.flush()
        sink.close()

        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            result = conn.execute(select(Table("dynamic_table", MetaData(), autoload_with=engine)))
            rows = result.fetchall()

        assert len(rows) == 1

    def test_flush_is_idempotent(self, test_db: str, ctx: PluginContext) -> None:
        """flush() can be called multiple times."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": test_db, "table": "results"})
        sink.write({"name": "test", "value": 1}, ctx)
        sink.flush()
        sink.flush()  # Should not raise
        sink.close()

    def test_close_without_write(self, test_db: str) -> None:
        """close() works even if no rows written."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        sink = DatabaseSink({"url": test_db, "table": "results"})
        sink.close()  # Should not raise
```

### Step 2: Run test to verify it fails

Run: `pytest tests/plugins/sinks/test_database_sink.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/plugins/sinks/database_sink.py
"""Database sink plugin for ELSPETH.

Writes rows to database tables using SQLAlchemy Core.
"""

from typing import Any

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
)
from sqlalchemy.engine import Engine

from elspeth.plugins.base import BaseSink
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schemas import PluginSchema


class DatabaseInputSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""

    model_config = {"extra": "allow"}


class DatabaseSink(BaseSink):
    """Write rows to a database table.

    Config options:
        url: Database URL (required) - e.g., "sqlite:///data.db"
        table: Table name (required)
        batch_size: Rows per commit (default: 100)
        auto_create: Create table from first row if missing (default: False)
    """

    name = "database"
    input_schema = DatabaseInputSchema
    idempotent = False  # INSERT can create duplicates

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._url = config["url"]
        self._table_name = config["table"]
        self._batch_size = config.get("batch_size", 100)
        self._auto_create = config.get("auto_create", False)

        self._engine: Engine | None = None
        self._table: Table | None = None
        self._metadata: MetaData | None = None
        self._buffer: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
        """Buffer a row for insertion."""
        if self._engine is None:
            self._engine = create_engine(self._url)
            self._metadata = MetaData()
            self._ensure_table(row)

        self._buffer.append(row)

        if len(self._buffer) >= self._batch_size:
            self._flush_buffer()

    def _ensure_table(self, sample_row: dict[str, Any]) -> None:
        """Ensure table exists, optionally creating it."""
        inspector = inspect(self._engine)

        if self._table_name in inspector.get_table_names():
            # Load existing table
            self._table = Table(
                self._table_name, self._metadata, autoload_with=self._engine
            )
        elif self._auto_create:
            # Create table from sample row
            columns = [self._infer_column(name, value) for name, value in sample_row.items()]
            self._table = Table(self._table_name, self._metadata, *columns)
            self._metadata.create_all(self._engine)
        else:
            raise ValueError(f"Table '{self._table_name}' does not exist and auto_create=False")

    def _infer_column(self, name: str, value: Any) -> Column:
        """Infer column type from Python value."""
        if isinstance(value, int):
            return Column(name, Integer)
        elif isinstance(value, float):
            return Column(name, Float)
        else:
            return Column(name, String(255))

    def _flush_buffer(self) -> None:
        """Insert buffered rows."""
        if not self._buffer or self._table is None or self._engine is None:
            return

        with self._engine.begin() as conn:
            conn.execute(self._table.insert(), self._buffer)

        self._buffer = []

    def flush(self) -> None:
        """Flush buffered rows to database."""
        self._flush_buffer()

    def close(self) -> None:
        """Close database connection."""
        self._flush_buffer()
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
```

Update sinks `__init__.py`:

```python
# src/elspeth/plugins/sinks/__init__.py
"""Built-in sink plugins for ELSPETH.

Sinks output data from the pipeline. One or more sinks per run.
"""

from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.database_sink import DatabaseSink
from elspeth.plugins.sinks.json_sink import JSONSink

__all__ = ["CSVSink", "DatabaseSink", "JSONSink"]
```

### Step 4: Run test to verify it passes

Run: `pytest tests/plugins/sinks/test_database_sink.py -v`
Expected: PASS (6 tests)

### Step 5: Commit

```bash
git add src/elspeth/plugins/sinks/ tests/plugins/sinks/test_database_sink.py
git commit -m "feat(plugins): add DatabaseSink plugin with SQLAlchemy Core"
```

---

## Task 7: CLI `plugins list` Command

**Context:** Implement the `elspeth plugins list` command to show available plugins.

**Files:**
- Modify: `src/elspeth/cli.py`
- Create: `tests/cli/test_plugins_command.py`

### Step 1: Write the failing test

```python
# tests/cli/test_plugins_command.py
"""Tests for elspeth plugins command."""

import pytest
from typer.testing import CliRunner

# IMPORTANT: mix_stderr=True ensures error messages are captured in stdout
# for consistent test assertions (Typer/Click writes errors to stderr by default)
runner = CliRunner(mix_stderr=True)


class TestPluginsListCommand:
    """Tests for plugins list command."""

    def test_plugins_list_shows_sources(self) -> None:
        """plugins list shows source plugins."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0
        assert "Sources:" in result.stdout
        assert "csv" in result.stdout
        assert "json" in result.stdout

    def test_plugins_list_shows_sinks(self) -> None:
        """plugins list shows sink plugins."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0
        assert "Sinks:" in result.stdout
        assert "csv" in result.stdout
        assert "json" in result.stdout
        assert "database" in result.stdout

    def test_plugins_list_type_filter(self) -> None:
        """plugins list --type filters by plugin type."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--type", "source"])
        assert result.exit_code == 0
        assert "Sources:" in result.stdout
        assert "Sinks:" not in result.stdout

    def test_plugins_list_json_output(self) -> None:
        """plugins list --json outputs JSON format."""
        import json

        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "sources" in data
        assert "sinks" in data
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_plugins_command.py -v`
Expected: FAIL (exit_code == 1, stub implementation)

### Step 3: Write implementation

Update `src/elspeth/cli.py` - replace the `plugins_list` stub:

```python
# Add imports at top of cli.py
import json as json_module
from typing import Optional

from elspeth.plugins.sources import CSVSource, JSONSource
from elspeth.plugins.sinks import CSVSink, JSONSink, DatabaseSink

# ... keep existing code ...

# Replace the plugins_list function:

@plugins_app.command("list")
def plugins_list(
    plugin_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by plugin type (source, transform, gate, aggregation, sink).",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON.",
    ),
) -> None:
    """List available plugins."""
    # NOTE: For Phase 4, we hard-code built-in plugins.
    # Future enhancement: Use PluginManager.get_sources(), etc.
    # when third-party plugin discovery is added.
    #
    # TODO(Phase 5+): Replace with:
    #   from elspeth.plugins.manager import PluginManager
    #   manager = PluginManager()
    #   manager.discover()  # Auto-discover registered plugins
    #   sources = manager.get_sources()
    plugins = {
        "sources": [
            {"name": CSVSource.name, "description": "Load rows from CSV files"},
            {"name": JSONSource.name, "description": "Load rows from JSON/JSONL files"},
        ],
        "transforms": [],
        "gates": [],
        "aggregations": [],
        "sinks": [
            {"name": CSVSink.name, "description": "Write rows to CSV files"},
            {"name": JSONSink.name, "description": "Write rows to JSON/JSONL files"},
            {"name": DatabaseSink.name, "description": "Write rows to database tables"},
        ],
    }

    # Filter by type if specified
    if plugin_type:
        type_key = plugin_type.lower() + "s" if not plugin_type.endswith("s") else plugin_type
        if type_key not in plugins:
            typer.echo(f"Unknown plugin type: {plugin_type}", err=True)
            raise typer.Exit(1)
        plugins = {type_key: plugins[type_key]}

    if json_output:
        typer.echo(json_module.dumps(plugins, indent=2))
    else:
        for category, items in plugins.items():
            if items:  # Only show non-empty categories
                typer.echo(f"\n{category.title()}:")
                for plugin in items:
                    typer.echo(f"  {plugin['name']:12} - {plugin['description']}")
        typer.echo()  # Trailing newline
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_plugins_command.py -v`
Expected: PASS (4 tests)

### Step 5: Commit

```bash
git add src/elspeth/cli.py tests/cli/test_plugins_command.py
git commit -m "feat(cli): implement plugins list command"
```

---

## Task 8: CLI `validate` Command

**Context:** Implement the `elspeth validate` command to validate pipeline configuration.

**Files:**
- Modify: `src/elspeth/cli.py`
- Create: `tests/cli/test_validate_command.py`

### Step 1: Write the failing test

```python
# tests/cli/test_validate_command.py
"""Tests for elspeth validate command."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

# IMPORTANT: mix_stderr=True ensures error messages are captured in stdout
# Without this, errors printed with err=True won't appear in result.stdout
runner = CliRunner(mix_stderr=True)


class TestValidateCommand:
    """Tests for validate command."""

    @pytest.fixture
    def valid_settings(self, tmp_path: Path) -> Path:
        """Create a valid settings file."""
        settings = {
            "source": {"plugin": "csv", "path": "data.csv"},
            "sinks": {"output": {"plugin": "json", "path": "output.json"}},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))
        return settings_file

    @pytest.fixture
    def invalid_settings(self, tmp_path: Path) -> Path:
        """Create an invalid settings file."""
        settings = {
            # Missing required 'source'
            "sinks": {"output": {"plugin": "json", "path": "output.json"}},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))
        return settings_file

    def test_validate_valid_config(self, valid_settings: Path) -> None:
        """validate exits 0 for valid config."""
        from elspeth.cli import app

        result = runner.invoke(app, ["validate", "--settings", str(valid_settings)])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_validate_invalid_config(self, invalid_settings: Path) -> None:
        """validate exits non-zero for invalid config."""
        from elspeth.cli import app

        result = runner.invoke(app, ["validate", "--settings", str(invalid_settings)])
        assert result.exit_code != 0
        assert "error" in result.stdout.lower() or "missing" in result.stdout.lower()

    def test_validate_missing_file(self) -> None:
        """validate exits non-zero for missing file."""
        from elspeth.cli import app

        result = runner.invoke(app, ["validate", "--settings", "/nonexistent/settings.yaml"])
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_validate_json_output(self, valid_settings: Path) -> None:
        """validate --json outputs JSON format."""
        import json

        from elspeth.cli import app

        result = runner.invoke(app, ["validate", "--settings", str(valid_settings), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["valid"] is True
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_validate_command.py -v`
Expected: FAIL (exit_code == 1, stub implementation)

### Step 3: Write implementation

Update `src/elspeth/cli.py` - add imports and replace `validate` stub:

```python
# Add imports
from pathlib import Path

import yaml

# Replace validate command:

@app.command()
def validate(
    settings: str = typer.Option(
        ...,
        "--settings",
        "-s",
        help="Path to settings YAML file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON.",
    ),
) -> None:
    """Validate pipeline configuration without running."""
    settings_path = Path(settings)

    # Check file exists
    if not settings_path.exists():
        if json_output:
            typer.echo(json_module.dumps({"valid": False, "error": f"File not found: {settings}"}))
        else:
            typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1)

    # Load and validate
    try:
        with open(settings_path) as f:
            config = yaml.safe_load(f)

        errors = _validate_config(config)

        if errors:
            if json_output:
                typer.echo(json_module.dumps({"valid": False, "errors": errors}))
            else:
                typer.echo("Configuration errors:", err=True)
                for error in errors:
                    typer.echo(f"  - {error}", err=True)
            raise typer.Exit(1)

        if json_output:
            typer.echo(json_module.dumps({"valid": True, "config": config}))
        else:
            typer.echo("Configuration is valid.")

    except yaml.YAMLError as e:
        if json_output:
            typer.echo(json_module.dumps({"valid": False, "error": f"YAML parse error: {e}"}))
        else:
            typer.echo(f"Error: Invalid YAML: {e}", err=True)
        raise typer.Exit(1)


def _validate_config(config: dict) -> list[str]:
    """Validate pipeline configuration, returning list of errors."""
    errors = []

    if not config:
        errors.append("Configuration is empty")
        return errors

    # Source is required
    if "source" not in config:
        errors.append("Missing required 'source' configuration")

    # At least one sink required
    if "sinks" not in config or not config.get("sinks"):
        errors.append("Missing required 'sinks' configuration")

    # Validate source structure
    if "source" in config:
        source = config["source"]
        if not isinstance(source, dict):
            errors.append("'source' must be a dictionary")
        elif "plugin" not in source:
            errors.append("Source missing required 'plugin' field")

    # Validate sinks structure
    if "sinks" in config and isinstance(config["sinks"], dict):
        for sink_name, sink_config in config["sinks"].items():
            if not isinstance(sink_config, dict):
                errors.append(f"Sink '{sink_name}' must be a dictionary")
            elif "plugin" not in sink_config:
                errors.append(f"Sink '{sink_name}' missing required 'plugin' field")

    return errors
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_validate_command.py -v`
Expected: PASS (4 tests)

### Step 5: Commit

```bash
git add src/elspeth/cli.py tests/cli/test_validate_command.py
git commit -m "feat(cli): implement validate command"
```

---

## Task 8.5: SinkAdapter Module

**Context:** Phase 3B's `SinkExecutor` expects sinks implementing the bulk `SinkLike` interface (`write(rows, ctx) -> dict`), but Phase 4 sinks implement the simpler Phase 2 row-wise interface (`write(row, ctx) -> None`). This task creates a proper adapter module to bridge the gap.

**Note:** Phase 3B has a **lifecycle gap** - Orchestrator closes source but never closes sinks. The adapter's `close()` method fills this gap, and the CLI must call it after `orchestrator.run()`.

**Files:**
- Create: `src/elspeth/engine/adapters.py`
- Create: `tests/engine/test_adapters.py`

### Step 1: Write the failing test

```python
# tests/engine/test_adapters.py
"""Tests for sink adapters."""

import hashlib
from pathlib import Path
from typing import Any

import pytest


class MockRowWiseSink:
    """Mock Phase 2 row-wise sink for testing."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.rows_written: list[dict] = []
        self.flushed = False
        self.closed = False

    def write(self, row: dict[str, Any], ctx: Any) -> None:
        self.rows_written.append(row)

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class TestSinkAdapter:
    """Tests for SinkAdapter."""

    def test_adapter_implements_bulk_write(self) -> None:
        """Adapter converts bulk write to row-wise writes."""
        from elspeth.engine.adapters import SinkAdapter

        mock_sink = MockRowWiseSink({"path": "/tmp/test.csv"})
        adapter = SinkAdapter(
            mock_sink,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        rows = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = adapter.write(rows, ctx=None)

        # Should have called write() for each row
        assert mock_sink.rows_written == rows
        # Should have flushed after writing
        assert mock_sink.flushed is True
        # Should NOT close in write() - lifecycle managed separately
        assert mock_sink.closed is False
        # Should return artifact info
        assert "kind" in result

    def test_adapter_close_calls_sink_close(self) -> None:
        """Adapter.close() closes the wrapped sink."""
        from elspeth.engine.adapters import SinkAdapter

        mock_sink = MockRowWiseSink({"path": "/tmp/test.csv"})
        adapter = SinkAdapter(
            mock_sink,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        adapter.close()
        assert mock_sink.closed is True

    def test_adapter_computes_artifact_hash(self, tmp_path: Path) -> None:
        """Adapter computes content hash from artifact file."""
        from elspeth.engine.adapters import SinkAdapter

        # Create a test file with known content
        test_file = tmp_path / "output.csv"
        test_content = b"id,name\n1,alice\n2,bob\n"
        test_file.write_bytes(test_content)
        expected_hash = hashlib.sha256(test_content).hexdigest()

        mock_sink = MockRowWiseSink({"path": str(test_file)})
        adapter = SinkAdapter(
            mock_sink,
            plugin_name="csv",
            sink_name="test",
            artifact_descriptor={"kind": "file", "path": str(test_file)},
        )

        # Test _compute_artifact_info directly to avoid write() side effects
        # (In production, write() modifies the file, changing the hash)
        result = adapter._compute_artifact_info()

        assert result["path"] == str(test_file)
        assert result["size_bytes"] == len(test_content)
        assert result["content_hash"] == expected_hash

    def test_adapter_write_then_hash(self, tmp_path: Path) -> None:
        """Adapter hashes file AFTER writes complete."""
        from elspeth.engine.adapters import SinkAdapter

        # Use a sink that actually writes to the file
        test_file = tmp_path / "output.csv"

        class RealWritingSink:
            def __init__(self, path: Path) -> None:
                self.config = {"path": str(path)}
                self._file = open(path, "w")
                self._file.write("id\n")  # Header

            def write(self, row: dict, ctx: Any) -> None:
                self._file.write(f"{row['id']}\n")

            def flush(self) -> None:
                self._file.flush()

            def close(self) -> None:
                self._file.close()

        real_sink = RealWritingSink(test_file)
        adapter = SinkAdapter(
            real_sink,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": str(test_file)},
        )

        # Write rows - this modifies the file
        result = adapter.write([{"id": 1}, {"id": 2}], ctx=None)
        adapter.close()

        # Hash should reflect FINAL file contents (header + 2 rows)
        expected_content = b"id\n1\n2\n"
        expected_hash = hashlib.sha256(expected_content).hexdigest()

        assert result["content_hash"] == expected_hash
        assert result["size_bytes"] == len(expected_content)

    def test_adapter_handles_missing_file(self) -> None:
        """Adapter handles non-existent artifact file gracefully."""
        from elspeth.engine.adapters import SinkAdapter

        mock_sink = MockRowWiseSink({"path": "/nonexistent/file.csv"})
        adapter = SinkAdapter(
            mock_sink,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/nonexistent/file.csv"},
        )

        result = adapter.write([{"id": 1}], ctx=None)

        assert result["path"] == "/nonexistent/file.csv"
        assert result["size_bytes"] == 0
        assert result["content_hash"] == ""

    def test_adapter_database_artifact(self) -> None:
        """Adapter handles database sinks with table-based artifact descriptors."""
        from elspeth.engine.adapters import SinkAdapter

        mock_sink = MockRowWiseSink({"url": "sqlite:///test.db", "table": "results"})
        adapter = SinkAdapter(
            mock_sink,
            plugin_name="database",
            sink_name="db_output",
            artifact_descriptor={
                "kind": "database",
                "url": "sqlite:///test.db",
                "table": "results",
            },
        )

        result = adapter.write([{"id": 1}], ctx=None)

        # Database artifacts don't have file hashes - descriptor is the identity
        assert result["kind"] == "database"
        assert result["table"] == "results"
        assert "content_hash" not in result or result["content_hash"] == ""

    def test_adapter_has_node_id_and_names(self) -> None:
        """Adapter exposes plugin_name, sink_name, and node_id for registration."""
        from elspeth.engine.adapters import SinkAdapter

        mock_sink = MockRowWiseSink({"path": "/tmp/test.csv"})
        adapter = SinkAdapter(
            mock_sink,
            plugin_name="csv",
            sink_name="flagged_output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        # plugin_name is the type (csv, json, database)
        assert adapter.plugin_name == "csv"
        # sink_name is the instance key from config
        assert adapter.sink_name == "flagged_output"
        # name property returns sink_name for SinkLike compatibility
        assert adapter.name == "flagged_output"
        # node_id starts empty, set by Orchestrator during registration
        assert hasattr(adapter, "node_id")
        adapter.node_id = "sink-001"
        assert adapter.node_id == "sink-001"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_adapters.py -v`
Expected: FAIL (ImportError)

### Step 3: Write implementation

```python
# src/elspeth/engine/adapters.py
"""Adapters for bridging Phase 2 plugins to Phase 3B engine interfaces.

Phase 2 plugins use simpler row-wise interfaces for ease of implementation.
Phase 3B engine expects bulk interfaces for efficiency and audit semantics.
These adapters bridge the gap.
"""

import hashlib
import os
from typing import Any, Protocol


class RowWiseSinkProtocol(Protocol):
    """Protocol for Phase 2 row-wise sinks."""

    config: dict[str, Any]

    def write(self, row: dict[str, Any], ctx: Any) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class SinkAdapter:
    """Adapts Phase 2 row-wise sinks to Phase 3B bulk SinkLike interface.

    Phase 2 SinkProtocol: write(row: dict, ctx) -> None (row-wise)
    Phase 3B SinkLike: write(rows: list[dict], ctx) -> dict (bulk, artifact info)

    Artifact Descriptors:
        Different sink types produce different artifact identities:
        - File sinks: {"kind": "file", "path": "output.csv"}
        - Database sinks: {"kind": "database", "url": "...", "table": "results"}
        - Webhook sinks: {"kind": "webhook", "url": "..."}

    Usage:
        raw_sink = CSVSink({"path": "output.csv"})
        adapter = SinkAdapter(
            raw_sink,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "output.csv"},
        )

        # Pass adapter to PipelineConfig (implements SinkLike)
        config = PipelineConfig(source=src, transforms=[], sinks={"output": adapter})

        # After orchestrator.run(), close the adapter (fills Phase 3B lifecycle gap)
        adapter.close()
    """

    def __init__(
        self,
        sink: RowWiseSinkProtocol,
        plugin_name: str,
        sink_name: str,
        artifact_descriptor: dict[str, Any],
    ) -> None:
        """Wrap a Phase 2 row-wise sink.

        Args:
            sink: Phase 2 sink implementing write(row, ctx) -> None
            plugin_name: Type of sink plugin (csv, json, database)
            sink_name: Instance name from config (output, flagged, etc.)
            artifact_descriptor: Describes the artifact identity by kind
        """
        self._sink = sink
        self.plugin_name = plugin_name
        self.sink_name = sink_name
        self.node_id: str = ""  # Set by Orchestrator during registration
        self._artifact_descriptor = artifact_descriptor

    @property
    def name(self) -> str:
        """Return sink_name for SinkLike protocol compatibility."""
        return self.sink_name

    def write(self, rows: list[dict[str, Any]], ctx: Any) -> dict[str, Any]:
        """Write rows using the wrapped sink's row-wise interface.

        Loops over rows, calling sink.write() for each, then flushes.
        Does NOT close the sink - close() must be called separately.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            Artifact info dict (structure depends on artifact kind)
        """
        # Loop over rows, calling Phase 2 row-wise write
        for row in rows:
            self._sink.write(row, ctx)

        # Flush buffered data (but don't close - lifecycle managed separately)
        self._sink.flush()

        # Compute artifact metadata based on descriptor kind
        return self._compute_artifact_info()

    def close(self) -> None:
        """Close the wrapped sink.

        Must be called after orchestrator.run() completes.
        Fills the Phase 3B lifecycle gap where Orchestrator doesn't close sinks.
        """
        self._sink.close()

    def _compute_artifact_info(self) -> dict[str, Any]:
        """Compute artifact metadata based on descriptor kind.

        Returns:
            Dict with artifact identity and optional content hash.
            Structure depends on kind:
            - file: {kind, path, size_bytes, content_hash}
            - database: {kind, url, table} (no content hash)
            - webhook: {kind, url} (no content hash)
        """
        kind = self._artifact_descriptor.get("kind", "unknown")

        if kind == "file":
            return self._compute_file_artifact()
        elif kind == "database":
            return self._compute_database_artifact()
        elif kind == "webhook":
            return self._compute_webhook_artifact()
        else:
            # Unknown kind - return descriptor as-is
            return dict(self._artifact_descriptor)

    def _compute_file_artifact(self) -> dict[str, Any]:
        """Compute artifact info for file-based sinks."""
        path = self._artifact_descriptor.get("path", "")
        size_bytes = 0
        content_hash = ""

        if path and os.path.exists(path):
            size_bytes = os.path.getsize(path)
            content_hash = self._hash_file_chunked(path)

        return {
            "kind": "file",
            "path": path,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
        }

    def _compute_database_artifact(self) -> dict[str, Any]:
        """Compute artifact info for database sinks.

        Database artifacts use the table identity, not content hashes.
        The audit trail links to the table; row-level integrity is the DB's job.
        """
        return {
            "kind": "database",
            "url": self._artifact_descriptor.get("url", ""),
            "table": self._artifact_descriptor.get("table", ""),
            # No content_hash - database is the source of truth
        }

    def _compute_webhook_artifact(self) -> dict[str, Any]:
        """Compute artifact info for webhook sinks."""
        return {
            "kind": "webhook",
            "url": self._artifact_descriptor.get("url", ""),
            # No content_hash - webhook response should be in calls table
        }

    @staticmethod
    def _hash_file_chunked(path: str, chunk_size: int = 65536) -> str:
        """Hash a file in chunks to avoid memory issues with large files.

        Args:
            path: Path to file
            chunk_size: Bytes to read per chunk (default 64KB)

        Returns:
            SHA-256 hex digest
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
```

### Step 4: Update engine __init__.py exports

Add to `src/elspeth/engine/__init__.py`:

```python
from elspeth.engine.adapters import SinkAdapter

__all__ = [
    # ... existing exports ...
    "SinkAdapter",
]
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/engine/test_adapters.py -v`
Expected: PASS (5 tests)

### Step 6: Commit

```bash
git add src/elspeth/engine/adapters.py tests/engine/test_adapters.py
git commit -m "feat(engine): add SinkAdapter for Phase 2/3B interface bridging"
```

---

## Task 9: CLI `run` Command

**Context:** Implement the `elspeth run` command to execute a pipeline. This integrates with the Phase 3 Orchestrator.

**Files:**
- Modify: `src/elspeth/cli.py`
- Create: `tests/cli/test_run_command.py`

### Step 1: Write the failing test

```python
# tests/cli/test_run_command.py
"""Tests for elspeth run command."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

# IMPORTANT: mix_stderr=True ensures error messages are captured in stdout
# Without this, errors printed with err=True won't appear in result.stdout
runner = CliRunner(mix_stderr=True)


class TestRunCommand:
    """Tests for run command."""

    @pytest.fixture
    def sample_data(self, tmp_path: Path) -> Path:
        """Create sample input data."""
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("id,name,value\n1,alice,100\n2,bob,200\n")
        return csv_file

    @pytest.fixture
    def pipeline_settings(self, tmp_path: Path, sample_data: Path) -> Path:
        """Create a complete pipeline configuration."""
        output_file = tmp_path / "output.json"
        landscape_db = tmp_path / "landscape.db"
        settings = {
            "source": {"plugin": "csv", "path": str(sample_data)},
            "sinks": {"output": {"plugin": "json", "path": str(output_file)}},
            # Use temp-path DB to avoid polluting CWD during tests
            "landscape": {"url": f"sqlite:///{landscape_db}"},
        }
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(yaml.dump(settings))
        return settings_file

    def test_run_executes_pipeline(
        self, pipeline_settings: Path, tmp_path: Path
    ) -> None:
        """run executes pipeline and creates output."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "--settings", str(pipeline_settings)])
        assert result.exit_code == 0

        # Check output was created
        output_file = tmp_path / "output.json"
        assert output_file.exists()

    def test_run_shows_summary(self, pipeline_settings: Path) -> None:
        """run shows execution summary."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "--settings", str(pipeline_settings)])
        assert result.exit_code == 0
        assert "completed" in result.stdout.lower() or "rows" in result.stdout.lower()

    def test_run_missing_settings(self) -> None:
        """run exits non-zero for missing settings file."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "--settings", "/nonexistent.yaml"])
        assert result.exit_code != 0

    def test_run_dry_run_mode(self, pipeline_settings: Path) -> None:
        """run --dry-run validates without executing."""
        from elspeth.cli import app

        result = runner.invoke(app, ["run", "--settings", str(pipeline_settings), "--dry-run"])
        assert result.exit_code == 0
        assert "dry" in result.stdout.lower() or "would" in result.stdout.lower()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_run_command.py -v`
Expected: FAIL (exit_code == 1, stub implementation)

### Step 3: Write implementation

Update `src/elspeth/cli.py` - replace `run` command:

```python
# Replace run command:

@app.command()
def run(
    settings: str = typer.Option(
        ...,
        "--settings",
        "-s",
        help="Path to settings YAML file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Validate and show what would run without executing.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output.",
    ),
) -> None:
    """Execute a pipeline run."""
    settings_path = Path(settings)

    # Check file exists
    if not settings_path.exists():
        typer.echo(f"Error: Settings file not found: {settings}", err=True)
        raise typer.Exit(1)

    # Load config
    try:
        with open(settings_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        typer.echo(f"Error: Invalid YAML: {e}", err=True)
        raise typer.Exit(1)

    # Validate config
    errors = _validate_config(config)
    if errors:
        typer.echo("Configuration errors:", err=True)
        for error in errors:
            typer.echo(f"  - {error}", err=True)
        raise typer.Exit(1)

    if dry_run:
        typer.echo("Dry run mode - would execute:")
        typer.echo(f"  Source: {config['source']['plugin']}")
        typer.echo(f"  Sinks: {', '.join(config['sinks'].keys())}")
        return

    # Execute pipeline
    try:
        result = _execute_pipeline(config, verbose=verbose)
        typer.echo(f"\nRun completed: {result['status']}")
        typer.echo(f"  Rows processed: {result['rows_processed']}")
        typer.echo(f"  Run ID: {result['run_id']}")
    except Exception as e:
        typer.echo(f"Error during pipeline execution: {e}", err=True)
        raise typer.Exit(1)


def _execute_pipeline(config: dict, verbose: bool = False) -> dict:
    """Execute a pipeline from configuration.

    Returns:
        Dict with run_id, status, rows_processed.
    """
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig
    from elspeth.engine.adapters import SinkAdapter

    # Instantiate source
    source_config = config["source"]
    source_plugin = source_config["plugin"]
    source_options = {k: v for k, v in source_config.items() if k != "plugin"}

    if source_plugin == "csv":
        source = CSVSource(source_options)
    elif source_plugin == "json":
        source = JSONSource(source_options)
    else:
        raise ValueError(f"Unknown source plugin: {source_plugin}")

    # Instantiate sinks and wrap in SinkAdapter for Phase 3B compatibility
    sinks: dict[str, SinkAdapter] = {}
    for sink_name, sink_config in config["sinks"].items():
        sink_plugin = sink_config["plugin"]
        sink_options = {k: v for k, v in sink_config.items() if k != "plugin"}

        if sink_plugin == "csv":
            raw_sink = CSVSink(sink_options)
            artifact_descriptor = {"kind": "file", "path": sink_options.get("path", "")}
        elif sink_plugin == "json":
            raw_sink = JSONSink(sink_options)
            artifact_descriptor = {"kind": "file", "path": sink_options.get("path", "")}
        elif sink_plugin == "database":
            raw_sink = DatabaseSink(sink_options)
            artifact_descriptor = {
                "kind": "database",
                "url": sink_options.get("url", ""),
                "table": sink_options.get("table", ""),
            }
        else:
            raise ValueError(f"Unknown sink plugin: {sink_plugin}")

        # Wrap Phase 2 sink in adapter for Phase 3B SinkLike interface
        sinks[sink_name] = SinkAdapter(
            raw_sink,
            plugin_name=sink_plugin,
            sink_name=sink_name,
            artifact_descriptor=artifact_descriptor,
        )

    # Get database URL from settings or use default
    db_url = config.get("landscape", {}).get("url", "sqlite:///elspeth_runs.db")
    db = LandscapeDB.from_url(db_url)

    # Build PipelineConfig
    pipeline_config = PipelineConfig(
        source=source,
        transforms=[],  # No transforms in basic Phase 4
        sinks=sinks,
    )

    if verbose:
        typer.echo("Starting pipeline execution...")

    # Execute via Orchestrator (creates full audit trail)
    # NOTE: Orchestrator takes LandscapeDB, not LandscapeRecorder - it creates its own recorder
    # NOTE: Orchestrator closes source and sinks in its finally block - no cleanup needed here
    orchestrator = Orchestrator(db)
    result = orchestrator.run(pipeline_config)

    return {
        "run_id": result.run_id,
        "status": result.status,
        "rows_processed": result.rows_processed,
    }
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_run_command.py -v`
Expected: PASS (4 tests)

### Step 5: Commit

```bash
git add src/elspeth/cli.py tests/cli/test_run_command.py
git commit -m "feat(cli): implement run command with basic pipeline execution"
```

---

## Task 10: Lineage Query Helper

**Context:** Create a helper function that composes LandscapeRecorder query methods to produce complete lineage for the explain command.

**Files:**
- Create: `src/elspeth/core/landscape/lineage.py`
- Create: `tests/core/landscape/test_lineage.py`

### Step 1: Write the failing test

```python
# tests/core/landscape/test_lineage.py
"""Tests for lineage query functionality."""

from datetime import datetime, timezone

import pytest


class TestLineageResult:
    """Tests for LineageResult data structure."""

    def test_lineage_result_exists(self) -> None:
        """LineageResult can be imported."""
        from elspeth.core.landscape.lineage import LineageResult

        assert LineageResult is not None

    def test_lineage_result_fields(self) -> None:
        """LineageResult has expected fields."""
        from elspeth.core.landscape.lineage import LineageResult
        from elspeth.core.landscape.models import Row, Token

        result = LineageResult(
            token=Token(
                token_id="t1",
                row_id="r1",
                created_at=datetime.now(timezone.utc),
            ),
            source_row=Row(
                row_id="r1",
                run_id="run1",
                source_node_id="src",
                row_index=0,
                source_data_hash="abc",
                created_at=datetime.now(timezone.utc),
            ),
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
        )
        assert result.token.token_id == "t1"
        assert result.source_row.row_id == "r1"


class TestExplainFunction:
    """Tests for explain() lineage query function."""

    def test_explain_exists(self) -> None:
        """explain function can be imported."""
        from elspeth.core.landscape.lineage import explain

        assert callable(explain)

    def test_explain_returns_lineage_result(self) -> None:
        """explain returns LineageResult."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.lineage import LineageResult, explain
        from elspeth.core.landscape.recorder import LandscapeRecorder

        # Setup: create a minimal run with one row
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="sha256-rfc8785-v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"id": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        recorder.complete_run(run.run_id, status="completed")

        # Query lineage
        result = explain(recorder, run_id=run.run_id, token_id=token.token_id)

        assert isinstance(result, LineageResult)
        assert result.token.token_id == token.token_id
        assert result.source_row.row_id == row.row_id

    def test_explain_by_row_id(self) -> None:
        """explain can query by row_id instead of token_id."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.lineage import explain
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="sha256-rfc8785-v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"id": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        recorder.complete_run(run.run_id, status="completed")

        # Query by row_id
        result = explain(recorder, run_id=run.run_id, row_id=row.row_id)

        assert result is not None
        assert result.source_row.row_id == row.row_id

    def test_explain_nonexistent_returns_none(self) -> None:
        """explain returns None for nonexistent token."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.lineage import explain
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        result = explain(recorder, run_id="fake", token_id="fake")
        assert result is None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_lineage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.core.landscape.lineage'`

### Step 3: Write minimal implementation

```python
# src/elspeth/core/landscape/lineage.py
"""Lineage query functionality for ELSPETH Landscape.

Provides the explain() function to compose query results into
complete lineage for a token or row.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from elspeth.core.landscape.models import (
    Call,
    NodeState,
    Row,
    RoutingEvent,
    Token,
)

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


@dataclass
class LineageResult:
    """Complete lineage for a token.

    Contains all information needed to explain how a row
    was processed through the pipeline.
    """

    token: Token
    """The token being explained."""

    source_row: Row
    """The original source row."""

    node_states: list[NodeState]
    """All node states visited by this token, in order."""

    routing_events: list[RoutingEvent]
    """All routing events for this token's states."""

    calls: list[Call]
    """All external calls made during processing."""

    parent_tokens: list[Token]
    """Parent tokens (for tokens created by fork/coalesce)."""


def explain(
    recorder: "LandscapeRecorder",
    run_id: str,
    token_id: str | None = None,
    row_id: str | None = None,
) -> LineageResult | None:
    """Query complete lineage for a token or row.

    Args:
        recorder: LandscapeRecorder with query methods.
        run_id: Run ID to query.
        token_id: Token ID for precise lineage (preferred).
        row_id: Row ID (will use first token for that row).

    Returns:
        LineageResult with complete lineage, or None if not found.

    Raises:
        ValueError: If neither token_id nor row_id provided.
    """
    if token_id is None and row_id is None:
        raise ValueError("Must provide either token_id or row_id")

    # Resolve token_id from row_id if needed
    if token_id is None:
        tokens = recorder.get_tokens(row_id)
        if not tokens:
            return None
        token_id = tokens[0].token_id

    # Get the token
    token = recorder.get_token(token_id)
    if token is None:
        return None

    # Get source row
    row = recorder.get_row(token.row_id)
    if row is None:
        return None

    # Get node states for this token
    node_states = recorder.get_node_states_for_token(token_id)
    node_states.sort(key=lambda s: s.step_index)

    # Get routing events for each state
    routing_events = []
    for state in node_states:
        events = recorder.get_routing_events(state.state_id)
        routing_events.extend(events)

    # Get external calls for each state
    calls = []
    for state in node_states:
        state_calls = recorder.get_calls(state.state_id)
        calls.extend(state_calls)

    # Get parent tokens
    parent_tokens = []
    parents = recorder.get_token_parents(token_id)
    for parent in parents:
        parent_token_list = recorder.get_tokens_by_id(parent.parent_token_id)
        if parent_token_list:
            parent_tokens.append(parent_token_list[0])

    return LineageResult(
        token=token,
        source_row=row,
        node_states=node_states,
        routing_events=routing_events,
        calls=calls,
        parent_tokens=parent_tokens,
    )
```

**Note:** This implementation assumes LandscapeRecorder has the following query methods from Phase 3A:
- `get_tokens(row_id)` - Get tokens for a row
- `get_tokens_by_id(token_id)` - Get token by ID
- `get_row(row_id)` - Get row by ID
- `get_node_states(token_id)` - Get node states for token
- `get_routing_events(state_id)` - Get routing events for state
- `get_calls(state_id)` - Get calls for state
- `get_token_parents(token_id)` - Get parent relationships

If any are missing from Phase 3A, they need to be added there.

### Step 4: Run test to verify it passes

Run: `pytest tests/core/landscape/test_lineage.py -v`
Expected: PASS (5 tests)

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/lineage.py tests/core/landscape/test_lineage.py
git commit -m "feat(landscape): add lineage query helper for explain command"
```

---

## Task 11: Textual TUI Foundation

**Context:** Create the Textual app structure for the explain TUI. This task sets up the app shell; Task 12 adds the lineage widgets.

**Files:**
- Create: `src/elspeth/tui/__init__.py`
- Create: `src/elspeth/tui/explain_app.py`
- Create: `tests/tui/__init__.py`
- Create: `tests/tui/test_explain_app.py`

### Step 1: Write the failing test

```python
# tests/tui/__init__.py
"""TUI tests."""

# tests/tui/test_explain_app.py
"""Tests for Explain TUI app."""

import pytest
from textual.pilot import Pilot


class TestExplainApp:
    """Tests for ExplainApp."""

    def test_app_exists(self) -> None:
        """ExplainApp can be imported."""
        from elspeth.tui.explain_app import ExplainApp

        assert ExplainApp is not None

    @pytest.mark.asyncio
    async def test_app_starts(self) -> None:
        """App can start and stop."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as pilot:
            assert app.is_running

    @pytest.mark.asyncio
    async def test_app_has_header(self) -> None:
        """App has a header with title."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as pilot:
            # Check for header widget
            from textual.widgets import Header
            header = app.query_one(Header)
            assert header is not None

    @pytest.mark.asyncio
    async def test_app_has_footer(self) -> None:
        """App has a footer with keybindings."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as pilot:
            from textual.widgets import Footer
            footer = app.query_one(Footer)
            assert footer is not None

    @pytest.mark.asyncio
    async def test_quit_keybinding(self) -> None:
        """q key quits the app."""
        from elspeth.tui.explain_app import ExplainApp

        app = ExplainApp()
        async with app.run_test() as pilot:
            await pilot.press("q")
            # App should exit
            assert not app.is_running
```

### Step 2: Run test to verify it fails

Run: `pytest tests/tui/test_explain_app.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/tui/__init__.py
"""ELSPETH Terminal User Interface.

Provides interactive TUI components using Textual.
"""

from elspeth.tui.explain_app import ExplainApp

__all__ = ["ExplainApp"]

# src/elspeth/tui/explain_app.py
"""Explain TUI application for ELSPETH.

Provides interactive lineage exploration.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static


class ExplainApp(App):
    """Interactive TUI for exploring run lineage.

    Displays lineage tree and allows drilling into node states,
    routing decisions, and external calls.
    """

    TITLE = "ELSPETH Explain"
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
    }

    #lineage-tree {
        height: 100%;
        border: solid green;
    }

    #detail-panel {
        height: 100%;
        border: solid blue;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("?", "help", "Help"),
    ]

    def __init__(
        self,
        run_id: str | None = None,
        token_id: str | None = None,
        row_id: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.run_id = run_id
        self.token_id = token_id
        self.row_id = row_id

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Static("Lineage Tree (placeholder)", id="lineage-tree")
        yield Static("Detail Panel (placeholder)", id="detail-panel")
        yield Footer()

    def action_refresh(self) -> None:
        """Refresh lineage data."""
        self.notify("Refreshing...")

    def action_help(self) -> None:
        """Show help."""
        self.notify("Press q to quit, arrow keys to navigate")
```

### Step 4: Run test to verify it passes

Run: `pytest tests/tui/test_explain_app.py -v`
Expected: PASS (5 tests)

### Step 5: Commit

```bash
git add src/elspeth/tui/ tests/tui/
git commit -m "feat(tui): add Textual ExplainApp foundation"
```

---

## Task 12: CLI `explain` Command with TUI

**Context:** Connect the explain command to the Textual TUI. Also add a `--no-tui` option for text output.

**Files:**
- Modify: `src/elspeth/cli.py`
- Create: `tests/cli/test_explain_command.py`

### Step 1: Write the failing test

```python
# tests/cli/test_explain_command.py
"""Tests for elspeth explain command."""

import pytest
from typer.testing import CliRunner

# IMPORTANT: mix_stderr=True ensures error messages are captured in stdout
# for consistent test assertions (Typer/Click writes errors to stderr by default)
runner = CliRunner(mix_stderr=True)


class TestExplainCommand:
    """Tests for explain command."""

    def test_explain_requires_run_id(self) -> None:
        """explain requires --run option."""
        from elspeth.cli import app

        result = runner.invoke(app, ["explain"])
        assert result.exit_code != 0
        assert "missing" in result.stdout.lower() or "required" in result.stdout.lower()

    def test_explain_no_tui_mode(self) -> None:
        """explain --no-tui outputs text instead of TUI."""
        from elspeth.cli import app

        # Note: This will fail gracefully since no runs exist
        result = runner.invoke(app, ["explain", "--run", "test-run", "--no-tui"])
        # Should not crash, may report "run not found"
        assert "error" in result.stdout.lower() or "not found" in result.stdout.lower()

    def test_explain_json_output(self) -> None:
        """explain --json outputs JSON format."""
        from elspeth.cli import app

        result = runner.invoke(app, ["explain", "--run", "test-run", "--json"])
        # Should output JSON (even if error)
        assert result.stdout.strip().startswith("{") or result.stdout.strip().startswith("[")
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_explain_command.py -v`
Expected: FAIL (exit_code == 1, stub implementation)

### Step 3: Write implementation

Update `src/elspeth/cli.py` - replace `explain` command:

```python
# Add import at top
from elspeth.tui.explain_app import ExplainApp

# Replace explain command:

@app.command()
def explain(
    run_id: str = typer.Option(
        ...,
        "--run",
        "-r",
        help="Run ID to explain (or 'latest').",
    ),
    row: Optional[str] = typer.Option(
        None,
        "--row",
        help="Row ID or index to explain.",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        help="Token ID for precise lineage.",
    ),
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        help="Output text instead of interactive TUI.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON.",
    ),
) -> None:
    """Explain lineage for a row or token."""
    # For now, we need a database connection to query
    # This will be integrated with actual runs once Phase 3 is complete

    if json_output:
        # JSON output mode
        result = {
            "run_id": run_id,
            "status": "error",
            "message": "No runs found. Execute 'elspeth run' first.",
        }
        typer.echo(json_module.dumps(result, indent=2))
        raise typer.Exit(1)

    if no_tui:
        # Text output mode
        typer.echo(f"Error: Run '{run_id}' not found.")
        typer.echo("Execute 'elspeth run' to create a run first.")
        raise typer.Exit(1)

    # TUI mode
    tui_app = ExplainApp(
        run_id=run_id if run_id != "latest" else None,
        token_id=token,
        row_id=row,
    )
    tui_app.run()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/cli/test_explain_command.py -v`
Expected: PASS (3 tests)

### Step 5: Commit

```bash
git add src/elspeth/cli.py tests/cli/test_explain_command.py
git commit -m "feat(cli): implement explain command with TUI and text modes"
```

---

## Task 13: structlog Integration

**Context:** Configure structlog for structured logging that complements OpenTelemetry spans.

**Files:**
- Create: `src/elspeth/core/logging.py`
- Create: `tests/core/test_logging.py`
- Modify: `src/elspeth/cli.py` (add logging setup)

### Step 1: Write the failing test

```python
# tests/core/test_logging.py
"""Tests for structured logging configuration."""

import json

import pytest


class TestLoggingConfig:
    """Tests for logging configuration."""

    def test_get_logger_exists(self) -> None:
        """get_logger function exists."""
        from elspeth.core.logging import get_logger

        assert callable(get_logger)

    def test_get_logger_returns_logger(self) -> None:
        """get_logger returns a bound logger."""
        from elspeth.core.logging import get_logger

        logger = get_logger("test")
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "bind")

    def test_configure_logging_exists(self) -> None:
        """configure_logging function exists."""
        from elspeth.core.logging import configure_logging

        assert callable(configure_logging)

    def test_logger_outputs_structured(self, capsys) -> None:
        """Logger outputs structured JSON."""
        from elspeth.core.logging import configure_logging, get_logger

        configure_logging(json_output=True)
        logger = get_logger("test")

        logger.info("test message", key="value")

        captured = capsys.readouterr()
        # Should be valid JSON
        log_line = captured.out.strip().split("\n")[-1]
        data = json.loads(log_line)
        assert data["event"] == "test message"
        assert data["key"] == "value"

    def test_logger_console_output(self, capsys) -> None:
        """Logger outputs human-readable in console mode."""
        from elspeth.core.logging import configure_logging, get_logger

        configure_logging(json_output=False)
        logger = get_logger("test")

        logger.info("test message", key="value")

        captured = capsys.readouterr()
        assert "test message" in captured.out
        # Should NOT be JSON
        assert not captured.out.strip().startswith("{")

    def test_logger_binds_context(self) -> None:
        """Logger can bind context."""
        from elspeth.core.logging import get_logger

        logger = get_logger("test")
        bound = logger.bind(run_id="abc123")

        assert bound is not None
        # Bound logger is a new instance
        assert bound is not logger
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_logging.py -v`
Expected: FAIL with `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/elspeth/core/logging.py
"""Structured logging configuration for ELSPETH.

Uses structlog for structured logging that complements
OpenTelemetry spans for observability.
"""

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    *,
    json_output: bool = False,
    level: str = "INFO",
) -> None:
    """Configure structlog for ELSPETH.

    Args:
        json_output: If True, output JSON. If False, human-readable.
        level: Log level (DEBUG, INFO, WARNING, ERROR).
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Shared processors
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        # JSON output for machine processing
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console output for humans
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        # IMPORTANT: Disable caching to allow reconfiguration in tests
        # Without this, tests that reconfigure logging will get stale loggers
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a bound logger for a module.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Bound structlog logger.
    """
    return structlog.get_logger(name)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/core/test_logging.py -v`
Expected: PASS (6 tests)

### Step 5: Commit

```bash
git add src/elspeth/core/logging.py tests/core/test_logging.py
git commit -m "feat(core): add structlog configuration"
```

---

## Task 14: Module Exports and Integration Test

**Context:** Update module `__init__.py` files to export new components and add an integration test.

**Files:**
- Modify: `src/elspeth/core/__init__.py`
- Modify: `src/elspeth/core/landscape/__init__.py`
- Modify: `src/elspeth/plugins/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_cli_integration.py`

### Step 1: Write the failing test

```python
# tests/integration/__init__.py
"""Integration tests."""

# tests/integration/test_cli_integration.py
"""Integration tests for CLI end-to-end workflow."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

# IMPORTANT: mix_stderr=True ensures error messages are captured in stdout
# Without this, errors printed with err=True won't appear in result.stdout
runner = CliRunner(mix_stderr=True)


class TestCLIIntegration:
    """End-to-end CLI integration tests."""

    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample CSV data."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("id,name,score\n1,alice,95\n2,bob,87\n3,carol,92\n")
        return csv_file

    @pytest.fixture
    def pipeline_config(self, tmp_path: Path, sample_csv: Path) -> Path:
        """Create pipeline configuration."""
        config = {
            "source": {"plugin": "csv", "path": str(sample_csv)},
            "sinks": {
                "json_output": {
                    "plugin": "json",
                    "path": str(tmp_path / "output.json"),
                },
                "csv_output": {
                    "plugin": "csv",
                    "path": str(tmp_path / "output.csv"),
                },
            },
            # Use temp-path DB to avoid polluting CWD during tests
            "landscape": {"url": f"sqlite:///{tmp_path / 'landscape.db'}"},
        }
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(yaml.dump(config))
        return config_file

    def test_full_workflow_csv_to_json(
        self, pipeline_config: Path, tmp_path: Path
    ) -> None:
        """Complete workflow: validate, run, check output."""
        from elspeth.cli import app

        # Step 1: Validate configuration
        result = runner.invoke(app, ["validate", "-s", str(pipeline_config)])
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

        # Step 2: Run pipeline
        result = runner.invoke(app, ["run", "-s", str(pipeline_config)])
        assert result.exit_code == 0
        assert "completed" in result.stdout.lower()

        # Step 3: Check output exists and is valid
        output_file = tmp_path / "output.json"
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert len(data) == 3
        assert data[0]["name"] == "alice"

    def test_plugins_list_shows_all_types(self) -> None:
        """plugins list shows sources and sinks."""
        from elspeth.cli import app

        result = runner.invoke(app, ["plugins", "list"])
        assert result.exit_code == 0

        # Sources
        assert "csv" in result.stdout
        assert "json" in result.stdout

        # Sinks
        assert "database" in result.stdout

    def test_dry_run_does_not_create_output(
        self, pipeline_config: Path, tmp_path: Path
    ) -> None:
        """dry-run does not create output files."""
        from elspeth.cli import app

        output_file = tmp_path / "output.json"
        assert not output_file.exists()

        result = runner.invoke(
            app, ["run", "-s", str(pipeline_config), "--dry-run"]
        )
        assert result.exit_code == 0

        # Output should NOT be created
        assert not output_file.exists()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/integration/test_cli_integration.py -v`
Expected: PASS (tests should work with existing implementation)

If tests fail, debug the integration issues.

### Step 3: Update module exports

```python
# src/elspeth/core/__init__.py
"""ELSPETH core infrastructure.

Foundational components: canonical hashing, configuration,
DAG validation, payload storage, logging.
"""

from elspeth.core.canonical import CANONICAL_VERSION, canonical_json, stable_hash
from elspeth.core.config import ElspethSettings, load_settings
from elspeth.core.dag import ExecutionGraph, NodeInfo
from elspeth.core.logging import configure_logging, get_logger
from elspeth.core.payload_store import FilesystemPayloadStore, PayloadStore

__all__ = [
    # Canonical
    "CANONICAL_VERSION",
    "canonical_json",
    "stable_hash",
    # Config
    "ElspethSettings",
    "load_settings",
    # DAG
    "ExecutionGraph",
    "NodeInfo",
    # Logging
    "configure_logging",
    "get_logger",
    # Payload Store
    "FilesystemPayloadStore",
    "PayloadStore",
]
```

```python
# Update src/elspeth/core/landscape/__init__.py to include lineage
# Add to existing exports:

from elspeth.core.landscape.lineage import LineageResult, explain

# Add to __all__:
# "LineageResult",
# "explain",
```

### Step 4: Run test to verify it passes

Run: `pytest tests/integration/test_cli_integration.py -v`
Expected: PASS (3 tests)

### Step 5: Commit

```bash
git add src/elspeth/core/__init__.py src/elspeth/core/landscape/__init__.py tests/integration/
git commit -m "feat: complete Phase 4 with module exports and integration tests"
```

---

## Summary

Phase 4 delivers:

1. **CLI Foundation** (Task 1) - Typer app with version, help, subcommands
2. **Source Plugins** (Tasks 2-3) - CSV and JSON sources
3. **Sink Plugins** (Tasks 4-6) - CSV, JSON, and Database sinks
4. **CLI Commands** (Tasks 7-9) - plugins list, validate, run
5. **Lineage Query** (Task 10) - explain() helper composing Landscape queries
6. **TUI Foundation** (Task 11) - Textual ExplainApp shell
7. **Explain Command** (Task 12) - CLI with TUI and text modes
8. **Structured Logging** (Task 13) - structlog configuration
9. **Integration** (Task 14) - Module exports and end-to-end tests

**Total: 14 tasks, ~60 test cases**

---

## Post-Phase 4 Status

After Phase 4, ELSPETH has:

- Working CLI (`elspeth run`, `elspeth validate`, `elspeth plugins list`)
- Basic I/O plugins (CSV, JSON sources; CSV, JSON, Database sinks)
- Explain foundation (TUI shell, lineage query helper)
- Structured logging with structlog

**Not Yet Complete:**
- Full TUI lineage visualization (tree widget, detail panels)
- Integration with Phase 3 Orchestrator (currently using simple loop)
- Transforms/Gates (no built-in transforms yet)
- Rate limiting, checkpointing (Phase 5)
- LLM integration (Phase 6)
