# WP-04: Sink Adapter Update

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update SinkAdapter to delegate directly to batch sinks, removing the per-row loop and capturing the sink's returned ArtifactDescriptor.

**Architecture:** The SinkAdapter currently wraps "Phase 2 row-wise sinks" with `write(row) -> None` signature and computes its own ArtifactDescriptor. After WP-03, sinks now have `write(rows) -> ArtifactDescriptor`. The adapter should detect which protocol the wrapped sink implements and either delegate directly (batch) or loop (row-wise, for backwards compatibility with any legacy sinks).

**Key Decision:** Keep the adapter layer rather than removing it entirely because:
1. The adapter provides `name`, `node_id`, and lifecycle management that the engine expects
2. Backwards compatibility with any row-wise sinks that may exist
3. Clean separation between sink plugins and engine integration

**Tech Stack:** Python 3.12, inspect.signature for sink type detection

---

## Task 1: Add is_batch_sink() detection function

**Files:**
- Modify: `src/elspeth/engine/adapters.py`
- Test: `tests/engine/test_adapters.py`

**Background:** Python's `@runtime_checkable` Protocol only checks that methods exist by name, NOT their signatures or return types. Both row-wise and batch sinks have `write`, `flush`, `close` methods, so `isinstance()` cannot distinguish them. Instead, we use **signature inspection** to check the parameter name (`row` vs `rows`).

**Step 1: Write the failing test**

Add test to verify sink type detection works:

```python
# Add to tests/engine/test_adapters.py after the imports

class TestSinkTypeDetection:
    """Tests for batch vs row-wise sink detection."""

    def test_is_batch_sink_exists(self) -> None:
        """is_batch_sink can be imported."""
        from elspeth.engine.adapters import is_batch_sink

        assert is_batch_sink is not None

    def test_detects_batch_sink(self) -> None:
        """Detects sink with write(rows: list) signature."""
        from typing import Any

        from elspeth.engine.adapters import is_batch_sink
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.plugins.context import PluginContext

        class BatchMockSink:
            """Mock sink with batch signature."""

            name = "batch_mock"

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                return ArtifactDescriptor.for_file(
                    path="/tmp/test.csv",
                    content_hash="abc123",
                    size_bytes=100,
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = BatchMockSink()
        assert is_batch_sink(sink) is True

    def test_detects_row_wise_sink(self) -> None:
        """Detects sink with write(row: dict) signature."""
        from typing import Any

        from elspeth.engine.adapters import is_batch_sink
        from elspeth.plugins.context import PluginContext

        class RowWiseMockSink:
            """Mock sink with row-wise signature."""

            name = "row_mock"

            def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
                pass

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = RowWiseMockSink()
        assert is_batch_sink(sink) is False

    def test_real_csv_sink_is_batch(self) -> None:
        """CSVSink (after WP-03) is detected as batch sink."""
        from elspeth.engine.adapters import is_batch_sink
        from elspeth.plugins.sinks.csv_sink import CSVSink

        sink = CSVSink({"path": "/tmp/test.csv"})
        assert is_batch_sink(sink) is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_adapters.py::TestSinkTypeDetection::test_is_batch_sink_exists -v`

Expected: FAIL with `ImportError: cannot import name 'is_batch_sink'`

**Step 3: Implement is_batch_sink()**

Add to `src/elspeth/engine/adapters.py` after the existing imports:

```python
import inspect

def is_batch_sink(sink: Any) -> bool:
    """Detect if a sink uses batch signature (write(rows)) vs row-wise (write(row)).

    Uses signature inspection to check the first parameter name of the write() method.
    Batch sinks have 'rows' (plural), row-wise sinks have 'row' (singular).

    This is necessary because Python's @runtime_checkable Protocol only checks
    that methods exist by name, not their signatures or return types.

    Args:
        sink: Sink instance to inspect

    Returns:
        True if sink has batch signature, False if row-wise
    """
    if not hasattr(sink, "write"):
        return False

    try:
        sig = inspect.signature(sink.write)
        params = list(sig.parameters.keys())
        # First param after 'self' (which is implicit for bound methods)
        # should be 'rows' for batch, 'row' for row-wise
        if params and params[0] == "rows":
            return True
        return False
    except (ValueError, TypeError):
        # Can't inspect signature, assume row-wise for safety
        return False
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_adapters.py::TestSinkTypeDetection -v`

Expected: All 4 tests pass

**Step 5: Commit**

```
git add -A && git commit -m "feat(adapters): add is_batch_sink() for sink type detection

Uses signature inspection to distinguish batch sinks (write(rows))
from row-wise sinks (write(row)). This is necessary because Python's
@runtime_checkable Protocol cannot check parameter names or return types.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Create BatchMockSink for testing

**Files:**
- Modify: `tests/engine/test_adapters.py`

**Step 1: Add BatchMockSink class**

Add after the existing `MockSink` class:

```python
class BatchMockSink:
    """Mock sink with batch signature for testing adapter delegation."""

    name = "batch_mock"

    def __init__(self) -> None:
        self.rows_written: list[dict[str, Any]] = []
        self._closed = False
        self._artifact_path = "/tmp/batch_mock_output.csv"

    def write(
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> "ArtifactDescriptor":
        from elspeth.contracts import ArtifactDescriptor

        self.rows_written.extend(rows)
        return ArtifactDescriptor.for_file(
            path=self._artifact_path,
            content_hash=f"hash_{len(self.rows_written)}",
            size_bytes=len(str(rows)),
        )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True
```

Add the import at the top of the file:

```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.contracts import ArtifactDescriptor
```

**Step 2: Commit**

```
git add -A && git commit -m "test(adapters): add BatchMockSink for batch delegation tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Update SinkAdapter to delegate to batch sinks

**Files:**
- Modify: `src/elspeth/engine/adapters.py`
- Test: `tests/engine/test_adapters.py`

**Step 1: Write failing tests for batch delegation**

Add to `tests/engine/test_adapters.py` in the `TestSinkAdapter` class:

```python
    def test_adapter_delegates_to_batch_sink(self, ctx: PluginContext) -> None:
        """Adapter delegates directly to batch sink without per-row loop."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.adapters import SinkAdapter

        batch_sink = BatchMockSink()
        adapter = SinkAdapter(
            batch_sink,
            plugin_name="batch_mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        rows = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = adapter.write(rows, ctx)

        # Batch sink receives all rows at once
        assert len(batch_sink.rows_written) == 3
        # Result is from sink, not computed by adapter
        assert isinstance(result, ArtifactDescriptor)
        assert result.content_hash == "hash_3"  # From BatchMockSink

    def test_adapter_uses_sink_artifact_for_batch(self, ctx: PluginContext) -> None:
        """Adapter returns sink's ArtifactDescriptor for batch sinks."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.adapters import SinkAdapter

        batch_sink = BatchMockSink()
        batch_sink._artifact_path = "/custom/path.csv"

        adapter = SinkAdapter(
            batch_sink,
            plugin_name="batch_mock",
            sink_name="output",
            # This artifact_descriptor should be IGNORED for batch sinks
            artifact_descriptor={"kind": "file", "path": "/different/path.csv"},
        )

        result = adapter.write([{"id": 1}], ctx)

        # Should use sink's path, not adapter's artifact_descriptor
        assert "/custom/path.csv" in result.path_or_uri

    def test_adapter_still_loops_for_row_wise_sink(self, ctx: PluginContext) -> None:
        """Adapter still uses per-row loop for row-wise sinks (backwards compat)."""
        from elspeth.engine.adapters import SinkAdapter

        row_sink = MockSink()  # Uses row-wise write(row, ctx) -> None
        adapter = SinkAdapter(
            row_sink,
            plugin_name="mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        rows = [{"id": 1}, {"id": 2}, {"id": 3}]
        adapter.write(rows, ctx)

        # Row-wise sink still receives rows individually
        assert len(row_sink.rows_written) == 3
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_adapters.py::TestSinkAdapter::test_adapter_delegates_to_batch_sink -v`

Expected: FAIL (adapter currently always loops)

**Step 3: Update SinkAdapter.write() to detect and delegate**

Modify the `write` method in `src/elspeth/engine/adapters.py`:

```python
    def write(self, rows: list[dict[str, Any]], ctx: Any) -> ArtifactDescriptor:
        """Write rows using the appropriate strategy based on sink type.

        For batch sinks (write(rows) -> ArtifactDescriptor): delegates directly
        For row-wise sinks (write(row) -> None): loops and computes artifact

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor from sink (batch) or computed (row-wise)
        """
        # Check if sink uses batch signature
        if is_batch_sink(self._sink):
            # Direct delegation - sink handles batching and returns artifact
            artifact = self._sink.write(rows, ctx)
            self._rows_written += len(rows)
            return artifact

        # Fallback: row-wise sink (backwards compatibility)
        # Store batch for hash computation
        self._last_batch_rows = list(rows)

        # Loop over rows, calling row-wise write
        for row in rows:
            self._sink.write(row, ctx)  # type: ignore[arg-type]
            self._rows_written += 1

        # Flush buffered data
        self._sink.flush()

        # Compute artifact metadata based on descriptor kind
        return self._compute_artifact_info()
```

Note: `is_batch_sink()` was added in Task 1. The existing `ArtifactDescriptor` import from `elspeth.engine.artifacts` is already present.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_adapters.py::TestSinkAdapter -v`

Expected: All tests pass, including the new delegation tests

**Step 5: Commit**

```
git add -A && git commit -m "feat(adapters): SinkAdapter delegates to batch sinks directly

SinkAdapter now detects BatchSinkProtocol and delegates write(rows)
directly to the sink, using the returned ArtifactDescriptor.

For row-wise sinks, the adapter still loops and computes its own
artifact (backwards compatibility).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Remove _rows_written dependency for batch sinks

**Files:**
- Modify: `src/elspeth/engine/adapters.py`
- Test: `tests/engine/test_adapters.py`

**Step 1: Write test to verify rows_written still works for batch sinks**

The `rows_written` property should still work, but be computed from the sink's data for batch sinks rather than being incremented per-row.

```python
    def test_rows_written_accurate_for_batch_sink(self, ctx: PluginContext) -> None:
        """rows_written is accurate after batch writes."""
        from elspeth.engine.adapters import SinkAdapter

        batch_sink = BatchMockSink()
        adapter = SinkAdapter(
            batch_sink,
            plugin_name="batch_mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        adapter.write([{"id": 1}, {"id": 2}], ctx)
        adapter.write([{"id": 3}], ctx)

        assert adapter.rows_written == 3
```

**Step 2: Run test**

This test should already pass because we increment `_rows_written += len(rows)` in the batch path.

Run: `pytest tests/engine/test_adapters.py::TestSinkAdapter::test_rows_written_accurate_for_batch_sink -v`

**Step 3: Commit (if any changes needed)**

If the test already passes, just add it and commit:

```
git add -A && git commit -m "test(adapters): verify rows_written accuracy for batch sinks

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Remove defensive hasattr patterns (Data Manifesto compliance)

**Files:**
- Modify: `src/elspeth/engine/adapters.py`

**Background:** The current adapter uses `hasattr` checks for `on_start` and `on_complete`:

```python
# CURRENT (WRONG) - treats plugin interface as untrusted
if hasattr(self._sink, "on_start"):
    self._sink.on_start(ctx)
```

This violates the **Data Manifesto**: plugins are system-owned code, not user data. If a sink doesn't have `on_start`, that's a bug in our code - we must crash, not silently skip.

**Step 1: Write failing test**

Add to `tests/engine/test_adapters.py`:

```python
class TestAdapterDataManifestoCompliance:
    """Tests verifying adapters don't hide plugin bugs."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return PluginContext(run_id="test-run", config={})

    def test_adapter_calls_on_start_unconditionally(self, ctx: PluginContext) -> None:
        """on_start() is called directly, not defensively checked."""
        from unittest.mock import MagicMock

        from elspeth.engine.adapters import SinkAdapter

        # Create a mock sink that tracks calls
        mock_sink = MagicMock()
        mock_sink.name = "mock"

        adapter = SinkAdapter(
            mock_sink,
            plugin_name="mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        adapter.on_start(ctx)

        # on_start MUST be called - no hasattr check
        mock_sink.on_start.assert_called_once_with(ctx)

    def test_adapter_calls_on_complete_unconditionally(self, ctx: PluginContext) -> None:
        """on_complete() is called directly, not defensively checked."""
        from unittest.mock import MagicMock

        from elspeth.engine.adapters import SinkAdapter

        mock_sink = MagicMock()
        mock_sink.name = "mock"

        adapter = SinkAdapter(
            mock_sink,
            plugin_name="mock",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        adapter.on_complete(ctx)

        mock_sink.on_complete.assert_called_once_with(ctx)

    def test_missing_on_start_crashes(self, ctx: PluginContext) -> None:
        """Sink without on_start causes AttributeError (not silent skip)."""
        from elspeth.engine.adapters import SinkAdapter

        class IncompleteStub:
            """Sink stub missing on_start - simulates plugin bug."""
            name = "incomplete"
            def write(self, rows, ctx): pass
            def flush(self): pass
            def close(self): pass
            # Intentionally missing: on_start, on_complete

        adapter = SinkAdapter(
            IncompleteStub(),
            plugin_name="incomplete",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "/tmp/test.csv"},
        )

        # MUST crash - plugin bug should not be hidden
        with pytest.raises(AttributeError):
            adapter.on_start(ctx)
```

**Step 2: Update adapter to remove hasattr**

Change `on_start` and `on_complete` in `adapters.py`:

```python
def on_start(self, ctx: Any) -> None:
    """Delegate on_start to wrapped sink.

    Per Data Manifesto: plugins are system-owned code.
    Missing on_start is a bug in the plugin - crash, don't hide.
    """
    self._sink.on_start(ctx)

def on_complete(self, ctx: Any) -> None:
    """Delegate on_complete to wrapped sink.

    Per Data Manifesto: plugins are system-owned code.
    Missing on_complete is a bug in the plugin - crash, don't hide.
    """
    self._sink.on_complete(ctx)
```

**Step 3: Run tests**

```bash
pytest tests/engine/test_adapters.py::TestAdapterDataManifestoCompliance -v
```

**Step 4: Commit**

```
git add -A && git commit -m "fix(adapters): remove defensive hasattr patterns per Data Manifesto

Plugins are system-owned code, not user data. If a sink is missing
on_start or on_complete, that's a bug in our code that must crash,
not be silently skipped.

BREAKING: Sinks MUST implement on_start() and on_complete().
This was always required by the protocol but not enforced.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Verify CLI sink creation (no changes needed)

**Files:**
- Review: `src/elspeth/cli.py` lines 284-292

**Step 1: Analyze CLI sink creation**

The CLI currently creates sinks and wraps them in SinkAdapter with explicit artifact_descriptor:

```python
sinks[sink_name] = SinkAdapter(
    raw_sink,
    plugin_name=sink_plugin,
    sink_name=sink_name,
    artifact_descriptor=artifact_descriptor,
)
```

For batch sinks, the `artifact_descriptor` is now redundant because the sink returns its own ArtifactDescriptor.

**Decision:** Keep the artifact_descriptor parameter for backwards compatibility and for cases where the adapter needs to compute artifacts for row-wise sinks. The adapter simply ignores it for batch sinks.

**No code changes needed** - the adapter handles this automatically.

**Step 2: Add integration test**

Add to `tests/engine/test_adapters.py`:

```python
class TestSinkAdapterIntegration:
    """Integration tests for SinkAdapter with real sink plugins."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        return PluginContext(run_id="test-run", config={})

    def test_adapter_with_csv_sink(self, ctx: PluginContext, tmp_path: Path) -> None:
        """SinkAdapter correctly delegates to CSVSink."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.adapters import SinkAdapter
        from elspeth.plugins.sinks.csv_sink import CSVSink

        output_file = tmp_path / "output.csv"
        sink = CSVSink({"path": str(output_file)})

        adapter = SinkAdapter(
            sink,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": str(output_file)},
        )

        rows = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = adapter.write(rows, ctx)

        # Verify delegation worked
        assert isinstance(result, ArtifactDescriptor)
        assert result.artifact_type == "file"
        assert result.size_bytes > 0
        assert len(result.content_hash) == 64  # SHA-256 hex

        # Verify file was written
        assert output_file.exists()
        content = output_file.read_text()
        assert "Alice" in content
        assert "Bob" in content

        sink.close()

    def test_adapter_with_json_sink(self, ctx: PluginContext, tmp_path: Path) -> None:
        """SinkAdapter correctly delegates to JSONSink."""
        import json

        from elspeth.contracts import ArtifactDescriptor
        from elspeth.engine.adapters import SinkAdapter
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_file = tmp_path / "output.json"
        sink = JSONSink({"path": str(output_file)})

        adapter = SinkAdapter(
            sink,
            plugin_name="json",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": str(output_file)},
        )

        rows = [{"id": 1, "value": "test"}]
        result = adapter.write(rows, ctx)

        assert isinstance(result, ArtifactDescriptor)
        assert result.artifact_type == "file"

        # Verify JSON content
        data = json.loads(output_file.read_text())
        assert data == rows

        sink.close()
```

**Step 3: Run integration tests**

Run: `pytest tests/engine/test_adapters.py::TestSinkAdapterIntegration -v`

**Step 4: Commit**

```
git add -A && git commit -m "test(adapters): add integration tests for SinkAdapter with real sinks

Verifies that SinkAdapter correctly delegates to CSVSink and JSONSink,
and that the returned ArtifactDescriptor contains valid content hashes.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Run full test suite and type checking

**Step 1: Run mypy**

```bash
mypy src/elspeth/engine/adapters.py --strict
```

Expected: No errors (or only pre-existing ones unrelated to this change)

**Step 2: Run all adapter tests**

```bash
pytest tests/engine/test_adapters.py -v
```

Expected: All tests pass

**Step 3: Run engine tests that use sinks**

```bash
pytest tests/engine/ -v -k "sink"
```

Expected: All tests pass

**Step 4: Final commit**

```
git add -A && git commit -m "chore: verify WP-04 sink adapter update complete

- SinkAdapter detects BatchSinkProtocol and delegates directly
- Row-wise sinks still work via per-row loop (backwards compat)
- Integration tests verify CSVSink and JSONSink delegation
- All type checks pass

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

- [ ] `is_batch_sink()` function exists and correctly detects sink types
- [ ] `is_batch_sink(CSVSink(...))` returns `True`
- [ ] `is_batch_sink(MockSink())` returns `False` (row-wise mock)
- [ ] `SinkAdapter.write()` delegates directly for batch sinks
- [ ] `SinkAdapter.write()` still loops for row-wise sinks (backwards compat)
- [ ] Adapter uses sink's returned `ArtifactDescriptor` for batch sinks
- [ ] Adapter computes own `ArtifactDescriptor` for row-wise sinks
- [ ] `rows_written` property is accurate for both sink types
- [ ] **Data Manifesto:** No `hasattr` checks for plugin methods
- [ ] **Data Manifesto:** `on_start()`/`on_complete()` called unconditionally
- [ ] **Data Manifesto:** Missing plugin method causes crash (not silent skip)
- [ ] Integration tests pass with real `CSVSink` and `JSONSink`
- [ ] `mypy --strict` passes on `adapters.py`
- [ ] All `tests/engine/test_adapters.py` tests pass

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/engine/adapters.py` | MODIFY | Add is_batch_sink(), update write(), remove hasattr |
| `tests/engine/test_adapters.py` | MODIFY | Add BatchMockSink, delegation tests, Data Manifesto compliance tests |

---

## Dependency Notes

- **Depends on:** WP-03 (sinks must have batch signatures)
- **Unlocks:** WP-13 (sink test rewrites can use batch patterns)
- **No changes needed to:** CLI, Orchestrator, Executors (adapter API unchanged)
