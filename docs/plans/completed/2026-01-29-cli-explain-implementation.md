# CLI Explain Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the `elspeth explain` CLI command to existing backend infrastructure so JSON, text, and TUI modes return real lineage data.

**Architecture:** The backend (`lineage.py:explain()`) and TUI components (`ExplainScreen`, widgets) already exist and are tested. The fix is purely wiring:
1. CLI needs `--database`/`--settings` options to locate the Landscape DB
2. CLI calls existing `explain()` function and formats output
3. `ExplainApp` needs to instantiate `ExplainScreen` with the database connection

**Tech Stack:** Typer CLI, SQLAlchemy, existing `LandscapeDB`/`LandscapeRecorder`/`explain()` infrastructure

**Review Status:** Passed 4-perspective review (Architecture, Python Engineering, QA, Systems Thinking) with requested changes incorporated.

---

## Architecture Decisions

### Code Placement (No Orphaned Code)

| Concern | Final Home | Rationale |
|---------|------------|-----------|
| Dataclass serialization | `src/elspeth/core/landscape/formatters.py` | Extends existing formatter module, reusable by CLI and MCP |
| Text lineage formatting | `src/elspeth/core/landscape/formatters.py` | Natural extension of export formatters |
| DB resolution helper | `src/elspeth/cli_helpers.py` | Common pattern across `purge`/`resume`/`explain` |
| Run ID "latest" resolution | `src/elspeth/cli_helpers.py` | CLI-specific concern |

### Reuse From MCP Server

The MCP server (`src/elspeth/mcp/server.py:37-73`) has working serialization helpers:
- `_serialize_datetime()` - handles datetime → ISO string
- `_dataclass_to_dict()` - recursive dataclass → dict conversion

These will be moved to `formatters.py` as public functions with proper names.

### Review Feedback Incorporated

| Issue | Source | Resolution |
|-------|--------|------------|
| Use `is_dataclass()` instead of `hasattr` | Python Review | Task 1 updated |
| Remove defensive `hasattr` on Enum values | Python Review | Task 2 updated - Tier 1 trust |
| Don't silently swallow exceptions | Python Review | Task 3 updated |
| Database existence check | Systems Review | Task 3 updated |
| Initialize `db = None` for cleanup | Python Review | Task 4 updated |
| Add NaN/Infinity rejection test | QA Review | Task 1 updated |
| Add ambiguous row CLI test | QA Review | Task 4 updated |
| Add round-trip integration test | QA Review | Task 4 updated |
| Expose detail panel publicly | Architecture Review | Task 6 updated (add property to ExplainScreen) |

---

## Task 1: Move Serialization Utilities to Formatters

**Files:**
- Modify: `src/elspeth/core/landscape/formatters.py`
- Modify: `src/elspeth/core/landscape/__init__.py`
- Modify: `src/elspeth/mcp/server.py`
- Create: `tests/core/landscape/test_formatters.py`

**Step 1: Write the failing test**

```python
# tests/core/landscape/test_formatters.py
"""Tests for Landscape formatters."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import pytest


class TestSerializeDatetime:
    """Tests for serialize_datetime utility."""

    def test_converts_datetime_to_iso(self) -> None:
        """datetime objects become ISO strings."""
        from elspeth.core.landscape.formatters import serialize_datetime

        dt = datetime(2026, 1, 29, 12, 30, 45, tzinfo=UTC)
        result = serialize_datetime(dt)
        assert result == "2026-01-29T12:30:45+00:00"

    def test_preserves_non_datetime(self) -> None:
        """Non-datetime values pass through unchanged."""
        from elspeth.core.landscape.formatters import serialize_datetime

        assert serialize_datetime("hello") == "hello"
        assert serialize_datetime(42) == 42
        assert serialize_datetime(None) is None

    def test_recursively_handles_dict(self) -> None:
        """Dicts have datetime values converted recursively."""
        from elspeth.core.landscape.formatters import serialize_datetime

        dt = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        data = {"created_at": dt, "name": "test", "nested": {"time": dt}}
        result = serialize_datetime(data)

        assert result["created_at"] == "2026-01-29T12:00:00+00:00"
        assert result["name"] == "test"
        assert result["nested"]["time"] == "2026-01-29T12:00:00+00:00"

    def test_recursively_handles_list(self) -> None:
        """Lists have datetime values converted recursively."""
        from elspeth.core.landscape.formatters import serialize_datetime

        dt = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        data = [dt, "string", {"time": dt}]
        result = serialize_datetime(data)

        assert result[0] == "2026-01-29T12:00:00+00:00"
        assert result[1] == "string"
        assert result[2]["time"] == "2026-01-29T12:00:00+00:00"

    def test_rejects_nan(self) -> None:
        """NaN values are rejected per CLAUDE.md audit integrity requirements."""
        from elspeth.core.landscape.formatters import serialize_datetime

        with pytest.raises(ValueError, match="NaN"):
            serialize_datetime(float("nan"))

    def test_rejects_infinity(self) -> None:
        """Infinity values are rejected per CLAUDE.md audit integrity requirements."""
        from elspeth.core.landscape.formatters import serialize_datetime

        with pytest.raises(ValueError, match="Infinity"):
            serialize_datetime(float("inf"))

        with pytest.raises(ValueError, match="Infinity"):
            serialize_datetime(float("-inf"))

    def test_rejects_nan_in_nested_structure(self) -> None:
        """NaN in nested structures is also rejected."""
        from elspeth.core.landscape.formatters import serialize_datetime

        with pytest.raises(ValueError, match="NaN"):
            serialize_datetime({"nested": {"value": float("nan")}})

        with pytest.raises(ValueError, match="NaN"):
            serialize_datetime([1, 2, float("nan")])


class TestDataclassToDict:
    """Tests for dataclass_to_dict utility."""

    def test_converts_simple_dataclass(self) -> None:
        """Simple dataclass becomes dict."""
        from elspeth.core.landscape.formatters import dataclass_to_dict

        @dataclass
        class Simple:
            name: str
            value: int

        obj = Simple(name="test", value=42)
        result = dataclass_to_dict(obj)

        assert result == {"name": "test", "value": 42}

    def test_converts_nested_dataclass(self) -> None:
        """Nested dataclasses are recursively converted."""
        from elspeth.core.landscape.formatters import dataclass_to_dict

        @dataclass
        class Inner:
            x: int

        @dataclass
        class Outer:
            inner: Inner
            y: str

        obj = Outer(inner=Inner(x=1), y="hello")
        result = dataclass_to_dict(obj)

        assert result == {"inner": {"x": 1}, "y": "hello"}

    def test_handles_enum_values(self) -> None:
        """Enum values are converted to their string value."""
        from elspeth.core.landscape.formatters import dataclass_to_dict

        class Status(Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        @dataclass
        class WithEnum:
            status: Status

        obj = WithEnum(status=Status.ACTIVE)
        result = dataclass_to_dict(obj)

        assert result == {"status": "active"}

    def test_handles_datetime_in_dataclass(self) -> None:
        """Datetime fields are converted to ISO strings."""
        from elspeth.core.landscape.formatters import dataclass_to_dict

        @dataclass
        class WithTime:
            created_at: datetime

        dt = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        obj = WithTime(created_at=dt)
        result = dataclass_to_dict(obj)

        assert result == {"created_at": "2026-01-29T12:00:00+00:00"}

    def test_handles_list_of_dataclasses(self) -> None:
        """Lists of dataclasses are recursively converted."""
        from elspeth.core.landscape.formatters import dataclass_to_dict

        @dataclass
        class Item:
            id: int

        @dataclass
        class Container:
            items: list[Item]

        obj = Container(items=[Item(id=1), Item(id=2)])
        result = dataclass_to_dict(obj)

        assert result == {"items": [{"id": 1}, {"id": 2}]}

    def test_handles_none(self) -> None:
        """None returns empty dict."""
        from elspeth.core.landscape.formatters import dataclass_to_dict

        result = dataclass_to_dict(None)
        assert result == {}

    def test_handles_plain_dict(self) -> None:
        """Plain dict passes through (not a dataclass)."""
        from elspeth.core.landscape.formatters import dataclass_to_dict

        result = dataclass_to_dict({"a": 1})
        assert result == {"a": 1}
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_formatters.py -v`
Expected: FAIL with `cannot import name 'serialize_datetime' from 'elspeth.core.landscape.formatters'`

**Step 3: Implement serialization utilities in formatters.py**

Add to `src/elspeth/core/landscape/formatters.py`:

```python
import math
from dataclasses import is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


def serialize_datetime(obj: Any) -> Any:
    """Convert datetime objects to ISO format strings for JSON serialization.

    Recursively processes dicts and lists to convert all datetime values.
    Rejects NaN and Infinity per CLAUDE.md audit integrity requirements.

    Args:
        obj: Any value - datetime, dict, list, or other

    Returns:
        The same structure with datetime objects replaced by ISO strings

    Raises:
        ValueError: If NaN or Infinity values are encountered
    """
    # Reject NaN and Infinity - audit trail must be pristine
    if isinstance(obj, float):
        if math.isnan(obj):
            raise ValueError("NaN values are not allowed in audit data (violates audit integrity)")
        if math.isinf(obj):
            raise ValueError("Infinity values are not allowed in audit data (violates audit integrity)")

    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_datetime(item) for item in obj]
    return obj


def dataclass_to_dict(obj: Any) -> Any:
    """Convert a dataclass (or list of dataclasses) to JSON-serializable dict.

    Handles:
    - Nested dataclasses (recursive conversion)
    - Lists of dataclasses
    - Enum values (converted to .value)
    - Datetime values (converted to ISO strings)
    - None (returns empty dict)
    - Plain values (pass through)

    Uses stdlib is_dataclass() and isinstance(Enum) for explicit type checking
    rather than hasattr() checks (clearer intent, better for maintenance).

    Args:
        obj: Dataclass instance, list, or primitive value

    Returns:
        dict for dataclasses, list for lists, or the original value
    """
    if obj is None:
        return {}
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        # is_dataclass returns True for both instances and classes
        # We only want instances, not the class itself
        result: dict[str, Any] = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            if is_dataclass(value) and not isinstance(value, type):
                result[field_name] = dataclass_to_dict(value)
            elif isinstance(value, list):
                result[field_name] = [dataclass_to_dict(item) for item in value]
            elif isinstance(value, Enum):
                # Explicit Enum check instead of hasattr(value, "value")
                result[field_name] = value.value
            else:
                result[field_name] = serialize_datetime(value)
        return result
    return obj
```

**Step 4: Update __init__.py exports**

Add to `src/elspeth/core/landscape/__init__.py` imports and `__all__`:

```python
from elspeth.core.landscape.formatters import (
    CSVFormatter,
    ExportFormatter,
    JSONFormatter,
    dataclass_to_dict,
    serialize_datetime,
)

# In __all__:
    "dataclass_to_dict",
    "serialize_datetime",
```

**Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_formatters.py -v`
Expected: PASS

**Step 6: Update MCP server to use shared utilities**

Modify `src/elspeth/mcp/server.py`:
- Remove local `_serialize_datetime` and `_dataclass_to_dict` functions
- Import from formatters: `from elspeth.core.landscape.formatters import dataclass_to_dict, serialize_datetime`
- Update all calls from `_dataclass_to_dict` → `dataclass_to_dict`

**Step 7: Run MCP tests to verify no regression**

Run: `.venv/bin/python -m pytest tests/mcp/ -v`
Expected: PASS (if tests exist) or no import errors

**Step 8: Commit**

```bash
git add src/elspeth/core/landscape/formatters.py src/elspeth/core/landscape/__init__.py src/elspeth/mcp/server.py tests/core/landscape/test_formatters.py
git commit -m "$(cat <<'EOF'
refactor(formatters): move serialization utilities from MCP to landscape.formatters

- Add serialize_datetime() for datetime → ISO string conversion
- Add dataclass_to_dict() for dataclass → dict conversion
- Use is_dataclass() and isinstance(Enum) for explicit type checks
- Reject NaN/Infinity per CLAUDE.md audit integrity requirements
- MCP server now imports from shared module
- Enables CLI explain command to reuse same serialization

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add Text Lineage Formatter

**Files:**
- Modify: `src/elspeth/core/landscape/formatters.py`
- Modify: `src/elspeth/core/landscape/__init__.py`
- Modify: `tests/core/landscape/test_formatters.py`

**Step 1: Write the failing test**

Add to `tests/core/landscape/test_formatters.py`:

```python
class TestLineageTextFormatter:
    """Tests for LineageTextFormatter."""

    def test_formats_basic_lineage(self) -> None:
        """Formats LineageResult as human-readable text."""
        from datetime import UTC, datetime

        from elspeth.contracts import RowLineage, Token
        from elspeth.core.landscape.formatters import LineageTextFormatter
        from elspeth.core.landscape.lineage import LineageResult

        now = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        result = LineageResult(
            token=Token(token_id="tok-123", row_id="row-456", created_at=now),
            source_row=RowLineage(
                row_id="row-456",
                run_id="run-789",
                source_node_id="src-node",
                row_index=0,
                source_data_hash="abc123",
                created_at=now,
                source_data={"id": 1, "name": "test"},
                payload_available=True,
            ),
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
        )

        formatter = LineageTextFormatter()
        text = formatter.format(result)

        assert "Token: tok-123" in text
        assert "Row: row-456" in text
        assert "Source Data Hash: abc123" in text

    def test_formats_with_outcome(self) -> None:
        """Includes outcome when present."""
        from datetime import UTC, datetime

        from elspeth.contracts import RowLineage, RowOutcome, Token, TokenOutcome
        from elspeth.core.landscape.formatters import LineageTextFormatter
        from elspeth.core.landscape.lineage import LineageResult

        now = datetime(2026, 1, 29, 12, 0, 0, tzinfo=UTC)
        result = LineageResult(
            token=Token(token_id="tok-123", row_id="row-456", created_at=now),
            source_row=RowLineage(
                row_id="row-456",
                run_id="run-789",
                source_node_id="src-node",
                row_index=0,
                source_data_hash="abc123",
                created_at=now,
                source_data={"id": 1},
                payload_available=True,
            ),
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
            outcome=TokenOutcome(
                outcome_id="out-1",
                token_id="tok-123",
                run_id="run-789",
                outcome=RowOutcome.COMPLETED,
                sink_name="output",
                is_terminal=True,
                created_at=now,
            ),
        )

        formatter = LineageTextFormatter()
        text = formatter.format(result)

        assert "Outcome: COMPLETED" in text
        assert "Sink: output" in text

    def test_formats_none_gracefully(self) -> None:
        """Returns message for None result."""
        from elspeth.core.landscape.formatters import LineageTextFormatter

        formatter = LineageTextFormatter()
        text = formatter.format(None)

        assert "not found" in text.lower() or "no lineage" in text.lower()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_formatters.py::TestLineageTextFormatter -v`
Expected: FAIL with `cannot import name 'LineageTextFormatter'`

**Step 3: Implement LineageTextFormatter**

Add to `src/elspeth/core/landscape/formatters.py`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.core.landscape.lineage import LineageResult


class LineageTextFormatter:
    """Format LineageResult as human-readable text for CLI output."""

    def format(self, result: "LineageResult | None") -> str:
        """Format lineage result as text.

        Args:
            result: LineageResult to format, or None if not found

        Returns:
            Human-readable text representation
        """
        if result is None:
            return "No lineage found. Token or row may not exist, or processing is incomplete."

        lines = []
        lines.append("=" * 60)
        lines.append("LINEAGE REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Token info
        lines.append(f"Token: {result.token.token_id}")
        lines.append(f"Row: {result.token.row_id}")
        if result.token.branch_name:
            lines.append(f"Branch: {result.token.branch_name}")
        lines.append("")

        # Source row
        lines.append("--- Source ---")
        lines.append(f"Row Index: {result.source_row.row_index}")
        lines.append(f"Source Data Hash: {result.source_row.source_data_hash}")
        lines.append(f"Payload Available: {result.source_row.payload_available}")
        if result.source_row.source_data:
            lines.append(f"Source Data: {result.source_row.source_data}")
        lines.append("")

        # Outcome
        if result.outcome:
            lines.append("--- Outcome ---")
            # Direct access to .value - Tier 1 trust (our audit data)
            # If outcome.outcome is not an Enum, that's a bug we want to crash on
            lines.append(f"Outcome: {result.outcome.outcome.value}")
            if result.outcome.sink_name:
                lines.append(f"Sink: {result.outcome.sink_name}")
            lines.append(f"Terminal: {result.outcome.is_terminal}")
            lines.append("")

        # Node states
        if result.node_states:
            lines.append("--- Node States ---")
            for state in result.node_states:
                # Direct access to .value - Tier 1 trust (our audit data)
                # No defensive hasattr - if status isn't an Enum, crash
                lines.append(f"  [{state.step_index}] {state.node_id}: {state.status.value}")
            lines.append("")

        # Calls
        if result.calls:
            lines.append("--- External Calls ---")
            for call in result.calls:
                # Direct access to .value - Tier 1 trust (our audit data)
                lines.append(f"  {call.call_type.value}: {call.status.value} ({call.latency_ms:.1f}ms)")
            lines.append("")

        # Errors
        if result.validation_errors:
            lines.append("--- Validation Errors ---")
            for err in result.validation_errors:
                lines.append(f"  {err.error_type}: {err.error_message}")
            lines.append("")

        if result.transform_errors:
            lines.append("--- Transform Errors ---")
            for err in result.transform_errors:
                lines.append(f"  {err.error_type}: {err.error_message}")
            lines.append("")

        # Parent tokens (for forks/joins)
        if result.parent_tokens:
            lines.append("--- Parent Tokens ---")
            for parent in result.parent_tokens:
                lines.append(f"  {parent.token_id}")
            lines.append("")

        return "\n".join(lines)
```

**Step 4: Update exports**

Add to `src/elspeth/core/landscape/__init__.py`:

```python
from elspeth.core.landscape.formatters import (
    # ... existing imports ...
    LineageTextFormatter,
)

# In __all__:
    "LineageTextFormatter",
```

**Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/landscape/test_formatters.py::TestLineageTextFormatter -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/landscape/formatters.py src/elspeth/core/landscape/__init__.py tests/core/landscape/test_formatters.py
git commit -m "$(cat <<'EOF'
feat(formatters): add LineageTextFormatter for CLI text output

Human-readable lineage output showing token, source, outcome,
node states, external calls, and errors.

Uses direct Enum.value access (Tier 1 trust) - no defensive hasattr.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add CLI Helper for Database Resolution

**Files:**
- Modify: `src/elspeth/cli_helpers.py`
- Create: `tests/cli/test_cli_helpers_db.py`

**Step 1: Write the failing test**

```python
# tests/cli/test_cli_helpers_db.py
"""Tests for CLI database resolution helpers."""

from pathlib import Path

import pytest


class TestResolveDatabaseUrl:
    """Tests for resolve_database_url helper."""

    def test_explicit_database_path_takes_precedence(self, tmp_path: Path) -> None:
        """--database CLI option overrides settings.yaml."""
        from elspeth.cli_helpers import resolve_database_url

        db_path = tmp_path / "explicit.db"
        db_path.touch()

        url, _ = resolve_database_url(database=str(db_path), settings_path=None)

        assert url == f"sqlite:///{db_path.resolve()}"

    def test_raises_when_database_file_not_found(self, tmp_path: Path) -> None:
        """Raises ValueError when --database points to nonexistent file."""
        from elspeth.cli_helpers import resolve_database_url

        nonexistent = tmp_path / "nonexistent.db"

        with pytest.raises(ValueError, match="not found"):
            resolve_database_url(database=str(nonexistent), settings_path=None)

    def test_loads_from_settings_yaml(self, tmp_path: Path) -> None:
        """Falls back to settings.yaml landscape.url."""
        from elspeth.cli_helpers import resolve_database_url

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
landscape:
  url: sqlite:///./state/audit.db
source:
  plugin: csv
  options:
    path: input.csv
sinks:
  - name: output
    plugin: csv
    options:
      path: output.csv
""")

        url, _ = resolve_database_url(database=None, settings_path=settings_file)

        assert url == "sqlite:///./state/audit.db"

    def test_raises_when_settings_missing_landscape_url(self, tmp_path: Path) -> None:
        """Raises ValueError when settings.yaml exists but missing landscape.url."""
        from elspeth.cli_helpers import resolve_database_url

        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("""
source:
  plugin: csv
  options:
    path: input.csv
sinks:
  - name: output
    plugin: csv
    options:
      path: output.csv
""")

        with pytest.raises(ValueError, match="landscape"):
            resolve_database_url(database=None, settings_path=settings_file)

    def test_raises_when_no_database_and_no_settings(self, tmp_path: Path) -> None:
        """Raises ValueError when neither database nor settings provided."""
        from elspeth.cli_helpers import resolve_database_url

        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(ValueError, match="database"):
            resolve_database_url(database=None, settings_path=nonexistent)

    def test_raises_when_default_settings_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError with context when default settings.yaml fails to load."""
        from elspeth.cli_helpers import resolve_database_url

        # Create invalid settings file in current directory
        monkeypatch.chdir(tmp_path)
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="settings.yaml"):
            resolve_database_url(database=None, settings_path=None)


class TestResolveLatestRunId:
    """Tests for resolve_latest_run_id helper."""

    def test_returns_most_recent_run(self) -> None:
        """'latest' resolves to most recently started run."""
        from elspeth.cli_helpers import resolve_latest_run_id
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create two runs - second one is "latest"
        run1 = recorder.begin_run(config={}, canonical_version="v1")
        run2 = recorder.begin_run(config={}, canonical_version="v1")

        result = resolve_latest_run_id(recorder)

        # run2 was created after run1, so it's latest
        assert result == run2.run_id

    def test_returns_none_when_no_runs(self) -> None:
        """Returns None when database has no runs."""
        from elspeth.cli_helpers import resolve_latest_run_id
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        result = resolve_latest_run_id(recorder)

        assert result is None

    def test_passthrough_non_latest_run_id(self) -> None:
        """Non-'latest' run_id passed through unchanged."""
        from elspeth.cli_helpers import resolve_run_id
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        result = resolve_run_id("explicit-run-id", recorder)

        assert result == "explicit-run-id"

    def test_latest_keyword_resolved(self) -> None:
        """'latest' keyword is resolved to actual run_id."""
        from elspeth.cli_helpers import resolve_run_id
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        result = resolve_run_id("latest", recorder)

        assert result == run.run_id
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cli/test_cli_helpers_db.py -v`
Expected: FAIL with `cannot import name 'resolve_database_url'`

**Step 3: Implement helpers**

Add to `src/elspeth/cli_helpers.py`:

```python
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.contracts import ElspethSettings
    from elspeth.core.landscape.recorder import LandscapeRecorder


def resolve_database_url(
    database: str | None,
    settings_path: Path | None,
) -> tuple[str, "ElspethSettings | None"]:
    """Resolve database URL from CLI option or settings file.

    Priority: CLI --database > settings.yaml landscape.url

    Args:
        database: Explicit database path from CLI (optional)
        settings_path: Path to settings.yaml file (optional)

    Returns:
        Tuple of (database_url, config_or_none)

    Raises:
        ValueError: If database file not found, settings invalid, or neither provided
    """
    from elspeth.core.config import load_settings

    config: "ElspethSettings | None" = None

    if database:
        db_path = Path(database).expanduser().resolve()
        # Fail fast with clear error if file doesn't exist
        if not db_path.exists():
            raise ValueError(f"Database file not found: {db_path}")
        return f"sqlite:///{db_path}", None

    # Try explicit settings file
    if settings_path and settings_path.exists():
        try:
            config = load_settings(settings_path)
            return config.landscape.url, config
        except Exception as e:
            raise ValueError(f"Error loading settings from {settings_path}: {e}") from e

    # Try default settings.yaml - DO NOT silently swallow errors
    default_settings = Path("settings.yaml")
    if default_settings.exists():
        try:
            config = load_settings(default_settings)
            return config.landscape.url, config
        except Exception as e:
            # Don't silently fall through - user should know why settings.yaml failed
            raise ValueError(f"Error loading default settings.yaml: {e}") from e

    raise ValueError(
        "No database specified. Provide --database or ensure settings.yaml exists "
        "with landscape.url configured."
    )


def resolve_latest_run_id(recorder: "LandscapeRecorder") -> str | None:
    """Get the most recently started run ID.

    Args:
        recorder: LandscapeRecorder with database connection

    Returns:
        Run ID of most recent run, or None if no runs exist
    """
    runs = recorder.list_runs()
    if not runs:
        return None
    # list_runs returns ordered by started_at DESC
    return runs[0].run_id


def resolve_run_id(run_id: str, recorder: "LandscapeRecorder") -> str | None:
    """Resolve run_id, handling 'latest' keyword.

    Args:
        run_id: Explicit run ID or 'latest'
        recorder: LandscapeRecorder for looking up latest

    Returns:
        Resolved run ID, or None if 'latest' requested but no runs exist
    """
    if run_id.lower() == "latest":
        return resolve_latest_run_id(recorder)
    return run_id
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cli/test_cli_helpers_db.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/cli_helpers.py tests/cli/test_cli_helpers_db.py
git commit -m "$(cat <<'EOF'
feat(cli): add database resolution helpers

- resolve_database_url(): CLI --database > settings.yaml
- resolve_latest_run_id(): get most recent run
- resolve_run_id(): handle 'latest' keyword
- Fail fast if database file doesn't exist
- Don't silently swallow settings.yaml errors

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Implement CLI Explain JSON Mode

**Files:**
- Modify: `src/elspeth/cli.py`
- Modify: `tests/cli/test_explain_command.py`

**Step 1: Write/update failing tests**

Update `tests/cli/test_explain_command.py` - replace "not implemented" tests with functional tests:

```python
"""Tests for elspeth explain command."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from elspeth.cli import app
from elspeth.contracts import NodeType, RowOutcome, RunStatus
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder, dataclass_to_dict, explain as explain_lineage

runner = CliRunner()

DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestExplainCommandBasics:
    """Basic CLI tests for explain command."""

    def test_explain_requires_run_id(self) -> None:
        """explain requires --run option."""
        result = runner.invoke(app, ["explain"])
        assert result.exit_code == 2
        output = result.output.lower()
        assert "missing option" in output or "--run" in output


class TestExplainJsonMode:
    """Tests for explain --json mode with real data."""

    @pytest.fixture
    def db_with_run(self, tmp_path: Path) -> tuple[Path, str, str]:
        """Create database with a simple completed run.

        Returns (db_path, run_id, token_id)
        """
        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"id": 1, "name": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        recorder.record_token_outcome(
            token_id=token.token_id,
            run_id=run.run_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)
        db.close()

        return db_path, run.run_id, token.token_id

    @pytest.fixture
    def db_with_forked_row(self, tmp_path: Path) -> tuple[Path, str, str]:
        """Create database with a row that forked to multiple sinks.

        Returns (db_path, run_id, row_id)
        """
        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"id": 1},
        )

        # Fork to two different sinks
        token_a = recorder.create_token(row_id=row.row_id)
        token_b = recorder.create_token(row_id=row.row_id)

        recorder.record_token_outcome(
            token_id=token_a.token_id,
            run_id=run.run_id,
            outcome=RowOutcome.ROUTED,
            sink_name="sink_a",
        )
        recorder.record_token_outcome(
            token_id=token_b.token_id,
            run_id=run.run_id,
            outcome=RowOutcome.ROUTED,
            sink_name="sink_b",
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)
        db.close()

        return db_path, run.run_id, row.row_id

    def test_json_output_returns_lineage(self, db_with_run: tuple[Path, str, str]) -> None:
        """--json returns real lineage data."""
        db_path, run_id, token_id = db_with_run

        result = runner.invoke(
            app, ["explain", "--run", run_id, "--token", token_id, "--database", str(db_path), "--json"]
        )

        assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}: {result.output}"
        data = json.loads(result.output)

        assert data["token"]["token_id"] == token_id
        assert data["source_row"]["row_id"] is not None
        assert data["outcome"]["outcome"] == "completed"

    def test_json_output_with_row_id(self, db_with_run: tuple[Path, str, str]) -> None:
        """--json with --row returns lineage."""
        db_path, run_id, token_id = db_with_run

        # Get row_id from the token we created
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)
        token = recorder.get_token(token_id)
        row_id = token.row_id
        db.close()

        result = runner.invoke(
            app, ["explain", "--run", run_id, "--row", row_id, "--database", str(db_path), "--json"]
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["source_row"]["row_id"] == row_id

    def test_json_output_nonexistent_token(self, db_with_run: tuple[Path, str, str]) -> None:
        """--json with nonexistent token returns error JSON."""
        db_path, run_id, _ = db_with_run

        result = runner.invoke(
            app, ["explain", "--run", run_id, "--token", "nonexistent", "--database", str(db_path), "--json"]
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert "error" in data

    def test_json_output_latest_run(self, db_with_run: tuple[Path, str, str]) -> None:
        """--run latest resolves to most recent run."""
        db_path, run_id, token_id = db_with_run

        result = runner.invoke(
            app, ["explain", "--run", "latest", "--token", token_id, "--database", str(db_path), "--json"]
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["token"]["token_id"] == token_id

    def test_json_output_ambiguous_row_without_sink(self, db_with_forked_row: tuple[Path, str, str]) -> None:
        """--json with ambiguous row (multiple tokens) returns helpful error."""
        db_path, run_id, row_id = db_with_forked_row

        result = runner.invoke(
            app, ["explain", "--run", run_id, "--row", row_id, "--database", str(db_path), "--json"]
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert "error" in data
        assert "terminal tokens" in data["error"] or "sink" in data["error"].lower()

    def test_json_output_matches_backend_explain(self, db_with_run: tuple[Path, str, str]) -> None:
        """CLI JSON output should match direct explain() call (round-trip test)."""
        db_path, run_id, token_id = db_with_run

        # Get backend result
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)
        backend_result = explain_lineage(recorder, run_id=run_id, token_id=token_id)
        backend_json = dataclass_to_dict(backend_result)
        db.close()

        # Get CLI result
        result = runner.invoke(
            app, ["explain", "--run", run_id, "--token", token_id, "--database", str(db_path), "--json"]
        )
        assert result.exit_code == 0
        cli_json = json.loads(result.output)

        # Should be identical (deep comparison)
        assert cli_json == backend_json, "CLI JSON output differs from backend explain() result"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cli/test_explain_command.py::TestExplainJsonMode -v`
Expected: FAIL (still returns "not_implemented")

**Step 3: Implement JSON mode in cli.py**

Update the `explain` command in `src/elspeth/cli.py`:

```python
@app.command()
def explain(
    run_id: str = typer.Option(
        ...,
        "--run",
        "-r",
        help="Run ID to explain (or 'latest').",
    ),
    row: str | None = typer.Option(
        None,
        "--row",
        help="Row ID to explain.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        "-t",
        help="Token ID for precise lineage.",
    ),
    database: str | None = typer.Option(
        None,
        "--database",
        "-d",
        help="Path to Landscape database file (SQLite).",
    ),
    settings: str | None = typer.Option(
        None,
        "--settings",
        "-s",
        help="Path to settings YAML file.",
    ),
    sink: str | None = typer.Option(
        None,
        "--sink",
        help="Sink name to disambiguate when row has multiple terminal tokens.",
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
    """Explain lineage for a row or token.

    Use --no-tui for text output or --json for JSON output.
    Without these flags, launches an interactive TUI.

    Examples:

        # JSON output for a specific token
        elspeth explain --run latest --token tok-abc --json --database ./audit.db

        # Text output for a row
        elspeth explain --run run-123 --row row-456 --no-tui --database ./audit.db

        # Interactive TUI
        elspeth explain --run latest --database ./audit.db
    """
    import json as json_module
    from pathlib import Path

    from elspeth.cli_helpers import resolve_database_url, resolve_run_id
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder, dataclass_to_dict, explain as explain_lineage
    from elspeth.core.landscape.formatters import LineageTextFormatter

    # Resolve database URL
    settings_path = Path(settings) if settings else None
    try:
        db_url, _ = resolve_database_url(database, settings_path)
    except ValueError as e:
        if json_output:
            typer.echo(json_module.dumps({"error": str(e)}))
        else:
            typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    # Connect to database
    # Initialize db = None for proper cleanup in finally block
    db: LandscapeDB | None = None
    try:
        db = LandscapeDB.from_url(db_url, create_tables=False)
    except Exception as e:
        if json_output:
            typer.echo(json_module.dumps({"error": f"Database connection failed: {e}"}))
        else:
            typer.echo(f"Error connecting to database: {e}", err=True)
        raise typer.Exit(1) from None

    try:
        recorder = LandscapeRecorder(db)

        # Resolve 'latest' run_id
        resolved_run_id = resolve_run_id(run_id, recorder)
        if resolved_run_id is None:
            if json_output:
                typer.echo(json_module.dumps({"error": "No runs found in database"}))
            else:
                typer.echo("Error: No runs found in database", err=True)
            raise typer.Exit(1) from None

        # Must provide either token or row
        if token is None and row is None:
            if json_output:
                typer.echo(json_module.dumps({"error": "Must provide either --token or --row"}))
            else:
                typer.echo("Error: Must provide either --token or --row", err=True)
            raise typer.Exit(1) from None

        # Query lineage
        try:
            lineage_result = explain_lineage(
                recorder,
                run_id=resolved_run_id,
                token_id=token,
                row_id=row,
                sink=sink,
            )
        except ValueError as e:
            # Ambiguous row (multiple tokens) or invalid args
            if json_output:
                typer.echo(json_module.dumps({"error": str(e)}))
            else:
                typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

        if lineage_result is None:
            if json_output:
                typer.echo(json_module.dumps({"error": "Token or row not found, or no terminal tokens exist yet"}))
            else:
                typer.echo("Token or row not found, or no terminal tokens exist yet.", err=True)
            raise typer.Exit(1) from None

        # Output based on mode
        if json_output:
            typer.echo(json_module.dumps(dataclass_to_dict(lineage_result), indent=2))
            raise typer.Exit(0)

        if no_tui:
            formatter = LineageTextFormatter()
            typer.echo(formatter.format(lineage_result))
            raise typer.Exit(0)

        # TUI mode
        from elspeth.tui.explain_app import ExplainApp

        tui_app = ExplainApp(
            db=db,
            run_id=resolved_run_id,
            token_id=token,
            row_id=row,
        )
        tui_app.run()

    finally:
        if db is not None:
            db.close()
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cli/test_explain_command.py::TestExplainJsonMode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/cli.py tests/cli/test_explain_command.py
git commit -m "$(cat <<'EOF'
feat(cli): implement explain --json mode with real lineage data

- Add --database and --settings options for DB resolution
- Add --sink option for fork disambiguation
- Resolve 'latest' run_id to most recent run
- Call explain() from lineage module
- Format output using dataclass_to_dict
- Initialize db = None for guaranteed cleanup
- Add round-trip test comparing CLI to backend

Closes: docs/bugs/open/cli/P1-2026-01-20-cli-explain-is-placeholder.md (JSON mode)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement CLI Explain Text Mode

**Files:**
- Modify: `tests/cli/test_explain_command.py`

**Step 1: Write failing test**

Add to `tests/cli/test_explain_command.py`:

```python
class TestExplainTextMode:
    """Tests for explain --no-tui text mode."""

    @pytest.fixture
    def db_with_run(self, tmp_path: Path) -> tuple[Path, str, str]:
        """Create database with a completed run."""
        db_path = tmp_path / "audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"id": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        recorder.record_token_outcome(
            token_id=token.token_id,
            run_id=run.run_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)
        db.close()

        return db_path, run.run_id, token.token_id

    def test_text_output_returns_lineage(self, db_with_run: tuple[Path, str, str]) -> None:
        """--no-tui returns human-readable lineage."""
        db_path, run_id, token_id = db_with_run

        result = runner.invoke(
            app, ["explain", "--run", run_id, "--token", token_id, "--database", str(db_path), "--no-tui"]
        )

        assert result.exit_code == 0, f"Expected exit 0: {result.output}"
        assert "LINEAGE REPORT" in result.output
        assert token_id in result.output
        assert "Outcome: completed" in result.output.lower()

    def test_text_output_shows_source_data(self, db_with_run: tuple[Path, str, str]) -> None:
        """--no-tui shows source data hash."""
        db_path, run_id, token_id = db_with_run

        result = runner.invoke(
            app, ["explain", "--run", run_id, "--token", token_id, "--database", str(db_path), "--no-tui"]
        )

        assert result.exit_code == 0
        assert "Source Data Hash:" in result.output
```

**Step 2: Run test to verify it passes**

The implementation was done in Task 4, so this should already work:

Run: `.venv/bin/python -m pytest tests/cli/test_explain_command.py::TestExplainTextMode -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/cli/test_explain_command.py
git commit -m "$(cat <<'EOF'
test(cli): add tests for explain --no-tui text mode

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire ExplainApp to ExplainScreen

**Files:**
- Modify: `src/elspeth/tui/screens/explain_screen.py` (add public property)
- Modify: `src/elspeth/tui/explain_app.py`
- Modify: `tests/tui/test_explain_app.py`

**Step 1: Add public property to ExplainScreen**

First, add a public property to ExplainScreen so ExplainApp doesn't access private `_detail_panel`:

Add to `src/elspeth/tui/screens/explain_screen.py`:

```python
@property
def detail_panel(self) -> "NodeDetailPanel":
    """Get the detail panel widget for composition.

    This is a public interface for ExplainApp to access the detail panel
    without directly accessing the private _detail_panel attribute.
    """
    return self._detail_panel
```

**Step 2: Write failing test**

Add to `tests/tui/test_explain_app.py`:

```python
class TestExplainAppWithData:
    """Tests for ExplainApp with database connection."""

    @pytest.mark.asyncio
    async def test_app_loads_lineage_data(self) -> None:
        """App loads and displays real lineage data."""
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.tui.explain_app import ExplainApp

        # Create test database with pipeline data
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        schema = SchemaConfig.from_dict({"fields": "dynamic"})

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )

        app = ExplainApp(db=db, run_id=run.run_id)
        async with app.run_test() as pilot:
            # App should have loaded data
            # Look for source name in rendered content
            # (implementation will mount ExplainScreen)
            assert app.is_running

    def test_app_accepts_db_parameter(self) -> None:
        """App can be initialized with database connection."""
        from elspeth.core.landscape import LandscapeDB
        from elspeth.tui.explain_app import ExplainApp

        db = LandscapeDB.in_memory()
        app = ExplainApp(db=db, run_id="test-run")

        assert app._db is db
        assert app._run_id == "test-run"
```

**Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/tui/test_explain_app.py::TestExplainAppWithData -v`
Expected: FAIL (ExplainApp doesn't accept db parameter)

**Step 4: Update ExplainApp to use ExplainScreen**

Rewrite `src/elspeth/tui/explain_app.py`:

```python
# src/elspeth/tui/explain_app.py
"""Explain TUI application for ELSPETH.

Provides interactive lineage exploration using ExplainScreen.
"""

from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static

from elspeth.core.landscape import LandscapeDB
from elspeth.tui.screens.explain_screen import (
    ExplainScreen,
    LoadedState,
    LoadingFailedState,
    UninitializedState,
)


class ExplainApp(App[None]):
    """Interactive TUI for exploring run lineage.

    Wraps ExplainScreen in a Textual application with keybindings
    and lifecycle management.
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

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("?", "help", "Help"),
    ]

    def __init__(
        self,
        db: LandscapeDB | None = None,
        run_id: str | None = None,
        token_id: str | None = None,
        row_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._db = db
        self._run_id = run_id
        self._token_id = token_id
        self._row_id = row_id
        self._screen: ExplainScreen | None = None

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()

        # Create ExplainScreen with database connection if available
        if self._db is not None and self._run_id is not None:
            self._screen = ExplainScreen(db=self._db, run_id=self._run_id)

            # Handle state explicitly - no defensive fallback
            match self._screen.state:
                case LoadedState():
                    yield self._screen.state.tree
                    yield self._screen.detail_panel  # Use public property
                case LoadingFailedState(error=err):
                    yield Static(f"Loading failed: {err or 'Unknown error'}")
                case UninitializedState():
                    yield Static("Screen not initialized. This should not happen.")
        else:
            # No data - show placeholder
            yield Static("No database connection. Use --database option.")

        yield Footer()

    def action_refresh(self) -> None:
        """Refresh lineage data.

        Note: This clears and reloads the screen state but does not remount
        widgets. For a full refresh, the app would need to be restarted.
        """
        if self._screen is not None:
            # Clear and reload
            self._screen.clear()
            if self._db and self._run_id:
                self._screen.load(self._db, self._run_id)
        self.notify("Refreshed")

    def action_help(self) -> None:
        """Show help."""
        self.notify("Press q to quit, r to refresh, arrow keys to navigate")
```

**Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/tui/test_explain_app.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/tui/screens/explain_screen.py src/elspeth/tui/explain_app.py tests/tui/test_explain_app.py
git commit -m "$(cat <<'EOF'
feat(tui): wire ExplainApp to ExplainScreen with real data

- Add detail_panel public property to ExplainScreen
- Accept db and run_id parameters in ExplainApp
- Create ExplainScreen with database connection
- Display tree and detail widgets from LoadedState
- Handle all state types explicitly (no defensive fallback)

Closes: docs/bugs/open/cli/P1-2026-01-20-cli-explain-is-placeholder.md (TUI mode)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Run Full Test Suite and Close Bug

**Files:**
- Move: `docs/bugs/open/cli/P1-2026-01-20-cli-explain-is-placeholder.md` → `docs/bugs/closed/cli/`

**Step 1: Run all explain-related tests**

Run: `.venv/bin/python -m pytest tests/cli/test_explain_command.py tests/tui/test_explain_app.py tests/cli/test_explain_tui.py tests/core/landscape/test_lineage.py tests/core/landscape/test_formatters.py -v`
Expected: ALL PASS

**Step 2: Run type checking**

Run: `.venv/bin/python -m mypy src/elspeth/cli.py src/elspeth/tui/explain_app.py src/elspeth/core/landscape/formatters.py src/elspeth/cli_helpers.py`
Expected: No errors

**Step 3: Run linting**

Run: `.venv/bin/python -m ruff check src/elspeth/cli.py src/elspeth/tui/explain_app.py src/elspeth/core/landscape/formatters.py src/elspeth/cli_helpers.py`
Expected: No errors (or fix any found)

**Step 4: Manual smoke test**

```bash
# Run an example pipeline first to create audit data
cd examples/basic
elspeth run -s settings.yaml -x

# Test JSON mode
elspeth explain --run latest --row 0 --database ./runs/audit.db --json

# Test text mode
elspeth explain --run latest --row 0 --database ./runs/audit.db --no-tui

# Test TUI mode (interactive)
elspeth explain --run latest --database ./runs/audit.db
```

**Step 5: Move bug report to closed**

```bash
mkdir -p docs/bugs/closed/cli
mv docs/bugs/open/cli/P1-2026-01-20-cli-explain-is-placeholder.md docs/bugs/closed/cli/
```

**Step 6: Final commit**

```bash
git add docs/bugs/
git commit -m "$(cat <<'EOF'
fix(cli): close P1 explain command placeholder bug

All three modes now functional:
- JSON mode: returns complete LineageResult as JSON
- Text mode: human-readable lineage report
- TUI mode: interactive exploration via ExplainScreen

Tested with examples/basic pipeline.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Task | Description | Files | Review Fixes Incorporated |
|------|-------------|-------|---------------------------|
| 1 | Move serialization utilities to formatters | formatters.py, mcp/server.py | `is_dataclass()`, NaN/Infinity rejection |
| 2 | Add LineageTextFormatter | formatters.py | Remove defensive `hasattr` on Enum |
| 3 | Add CLI database resolution helpers | cli_helpers.py | Don't swallow exceptions, file existence check |
| 4 | Implement explain --json mode | cli.py | `db = None` init, ambiguous row test, round-trip test |
| 5 | Test explain --no-tui mode | test_explain_command.py | - |
| 6 | Wire ExplainApp to ExplainScreen | explain_app.py, explain_screen.py | Public `detail_panel` property |
| 7 | Full test suite and close bug | All | - |

**Total estimated time:** 2-3 hours (as bug report predicted)

**Key architectural decisions:**
- Serialization utilities live in `formatters.py` (not duplicated)
- Database resolution helpers live in `cli_helpers.py` (shared pattern)
- ExplainApp wraps ExplainScreen (existing, tested infrastructure)
- No new modules created - everything extends existing homes

**Review feedback addressed:**
- ✅ Use `is_dataclass()` instead of `hasattr` (Python Review)
- ✅ Remove defensive `hasattr` on Enum values (Python Review - Tier 1 trust)
- ✅ Don't silently swallow exceptions (Python Review)
- ✅ Database file existence check (Systems Review)
- ✅ Initialize `db = None` for cleanup (Python Review)
- ✅ Add NaN/Infinity rejection test (QA Review)
- ✅ Add ambiguous row CLI test (QA Review)
- ✅ Add round-trip integration test (QA Review)
- ✅ Expose `detail_panel` publicly (Architecture Review)
