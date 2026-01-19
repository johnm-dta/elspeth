# WP-01: Protocol & Base Class Alignment

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align SourceProtocol and SinkProtocol with contract v1.1 by adding missing metadata attributes and updating the sink write signature.

**Architecture:** This is a surgical change to two files (protocols.py and base.py). We add `determinism` and `plugin_version` to SourceProtocol (the only protocol missing these), then update the Sink.write() signature from per-row to batch mode. The actual sink implementations (CSVSink, JSONSink, etc.) are NOT updated here - they will fail type checking until WP-03 fixes them.

**Tech Stack:** Python 3.12, typing.Protocol, runtime_checkable decorators, pydantic schemas

---

## Task 1: Add determinism and plugin_version to SourceProtocol

**Files:**
- Modify: `src/elspeth/plugins/protocols.py:52-54`
- Test: `tests/plugins/test_protocols.py`

**Step 1: Write the failing test**

Add test to verify SourceProtocol has the required attributes:

```python
# Add to tests/plugins/test_protocols.py at the end of TestSourceProtocol class

    def test_source_has_determinism_attribute(self) -> None:
        from elspeth.plugins.protocols import SourceProtocol

        assert "determinism" in SourceProtocol.__protocol_attrs__

    def test_source_has_version_attribute(self) -> None:
        from elspeth.plugins.protocols import SourceProtocol

        assert "plugin_version" in SourceProtocol.__protocol_attrs__

    def test_source_implementation_with_metadata(self) -> None:
        from collections.abc import Iterator
        from typing import Any

        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SourceProtocol

        class OutputSchema(PluginSchema):
            value: int

        class MetadataSource:
            name = "metadata_source"
            output_schema = OutputSchema
            node_id: str | None = None
            determinism = Determinism.IO_READ
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                self.config = config

            def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
                yield {"value": 1}

            def close(self) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        source = MetadataSource({})
        assert isinstance(source, SourceProtocol)
        assert source.determinism == Determinism.IO_READ
        assert source.plugin_version == "1.0.0"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_protocols.py::TestSourceProtocol::test_source_has_determinism_attribute -v`

Expected: FAIL with `KeyError: 'determinism'` (attribute not in protocol)

**Step 3: Add determinism and plugin_version to SourceProtocol**

Edit `src/elspeth/plugins/protocols.py` lines 52-54. After `node_id`:

```python
    name: str
    output_schema: type["PluginSchema"]
    node_id: str | None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism
    plugin_version: str
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/test_protocols.py::TestSourceProtocol -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/protocols.py tests/plugins/test_protocols.py
git commit -m "$(cat <<'EOF'
feat(protocols): add determinism and plugin_version to SourceProtocol

SourceProtocol was the only protocol missing these metadata attributes
required by plugin-protocol contract v1.1.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add determinism and plugin_version to BaseSource

**Files:**
- Modify: `src/elspeth/plugins/base.py:321-324`
- Test: `tests/plugins/test_base.py`

**Step 1: Write the failing test**

Add test to verify BaseSource has metadata:

```python
# Add to tests/plugins/test_base.py at the end of TestBaseSource class

    def test_base_source_has_metadata_attributes(self) -> None:
        from elspeth.contracts import Determinism
        from elspeth.plugins.base import BaseSource

        assert hasattr(BaseSource, "determinism")
        assert hasattr(BaseSource, "plugin_version")
        # Check default values
        assert BaseSource.determinism == Determinism.IO_READ
        assert BaseSource.plugin_version == "0.0.0"

    def test_subclass_can_override_metadata(self) -> None:
        from collections.abc import Iterator
        from typing import Any

        from elspeth.contracts import Determinism, PluginSchema
        from elspeth.plugins.base import BaseSource
        from elspeth.plugins.context import PluginContext

        class OutputSchema(PluginSchema):
            value: int

        class CustomSource(BaseSource):
            name = "custom"
            output_schema = OutputSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "2.0.0"

            def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
                yield {"value": 1}

            def close(self) -> None:
                pass

        source = CustomSource({})
        assert source.determinism == Determinism.DETERMINISTIC
        assert source.plugin_version == "2.0.0"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_base.py::TestBaseSource::test_base_source_has_metadata_attributes -v`

Expected: FAIL with `AssertionError` (determinism not defined)

**Step 3: Add determinism and plugin_version to BaseSource**

Edit `src/elspeth/plugins/base.py` lines 321-324. After `node_id`:

```python
    name: str
    output_schema: type[PluginSchema]
    node_id: str | None = None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism = Determinism.IO_READ
    plugin_version: str = "0.0.0"
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/test_base.py::TestBaseSource -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/base.py tests/plugins/test_base.py
git commit -m "$(cat <<'EOF'
feat(base): add determinism and plugin_version to BaseSource

Defaults to IO_READ (sources read from external systems) and "0.0.0".
Subclasses can override these values.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update SinkProtocol.write() signature to batch mode

**Files:**
- Modify: `src/elspeth/plugins/protocols.py:480-490`
- Test: `tests/plugins/test_protocols.py`

**Step 1: Write the failing test**

Add test for new batch signature with ArtifactDescriptor return:

```python
# Add to tests/plugins/test_protocols.py in TestSinkProtocol class

    def test_sink_batch_write_signature(self) -> None:
        """Sink.write() accepts batch and returns ArtifactDescriptor."""
        import inspect
        from elspeth.plugins.protocols import SinkProtocol

        # Get the write method signature
        sig = inspect.signature(SinkProtocol.write)
        params = list(sig.parameters.keys())

        # Should have 'rows' not 'row'
        assert "rows" in params, "write() should accept 'rows' (batch), not 'row'"
        assert "row" not in params, "write() should NOT have 'row' parameter"

        # Return annotation should be ArtifactDescriptor
        from elspeth.contracts import ArtifactDescriptor
        assert sig.return_annotation == ArtifactDescriptor

    def test_batch_sink_implementation(self) -> None:
        """Test sink with batch write returning ArtifactDescriptor."""
        from typing import Any

        from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SinkProtocol

        class InputSchema(PluginSchema):
            value: int

        class BatchMemorySink:
            name = "batch_memory"
            input_schema = InputSchema
            idempotent = True
            node_id: str | None = None
            determinism = Determinism.IO_WRITE
            plugin_version = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                self.rows: list[dict[str, Any]] = []

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                self.rows.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="/tmp/test.json",
                    content_hash="abc123",
                    size_bytes=len(str(rows)),
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        sink = BatchMemorySink({})
        assert isinstance(sink, SinkProtocol)

        ctx = PluginContext(run_id="test", config={})
        artifact = sink.write([{"value": 1}, {"value": 2}], ctx)

        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.content_hash == "abc123"
        assert len(sink.rows) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_protocols.py::TestSinkProtocol::test_sink_batch_write_signature -v`

Expected: FAIL with `AssertionError: write() should accept 'rows' (batch), not 'row'`

**Step 3: Update SinkProtocol.write() signature**

Edit `src/elspeth/plugins/protocols.py` lines 454-490. Update the docstring example and write() method:

```python
@runtime_checkable
class SinkProtocol(Protocol):
    """Protocol for sink plugins.

    Sinks output data to external destinations.
    There can be multiple sinks per run.

    Idempotency:
    - Sinks receive idempotency keys: {run_id}:{row_id}:{sink_name}
    - Sinks that cannot guarantee idempotency should set idempotent=False

    Example:
        class CSVSink:
            name = "csv"
            input_schema = RowSchema
            idempotent = False  # Appends are not idempotent

            def write(self, rows: list[dict], ctx: PluginContext) -> ArtifactDescriptor:
                for row in rows:
                    self._writer.writerow(row)
                return ArtifactDescriptor.for_file(
                    path=self._path,
                    content_hash=self._compute_hash(),
                    size_bytes=self._file.tell(),
                )

            def flush(self) -> None:
                self._file.flush()
    """

    name: str
    input_schema: type["PluginSchema"]
    idempotent: bool  # Can this sink handle retries safely?
    node_id: str | None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism
    plugin_version: str

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        ...

    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: "PluginContext",
    ) -> "ArtifactDescriptor":
        """Write a batch of rows to the sink.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash and size_bytes (REQUIRED for audit)
        """
        ...

    def flush(self) -> None:
        """Flush any buffered data.

        Called periodically and at end of run.
        """
        ...

    def close(self) -> None:
        """Close the sink and release resources.

        Called at end of run or on error.
        """
        ...

    # === Optional Lifecycle Hooks ===

    def on_register(self, ctx: "PluginContext") -> None:
        """Called when plugin is registered."""
        ...

    def on_start(self, ctx: "PluginContext") -> None:
        """Called at start of run."""
        ...

    def on_complete(self, ctx: "PluginContext") -> None:
        """Called at end of run (before close)."""
        ...
```

Also add the import at the top of the file (if not already present):

```python
if TYPE_CHECKING:
    from elspeth.contracts import ArtifactDescriptor, PluginSchema
    # ... other imports
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/test_protocols.py::TestSinkProtocol -v`

Expected: New tests PASS. Old test (`test_sink_implementation`) will FAIL because it uses old signature.

**Step 5: Update old test to use new signature**

Update `test_sink_implementation` in TestSinkProtocol to use batch signature:

```python
    def test_sink_implementation(self) -> None:
        from typing import Any, ClassVar

        from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.protocols import SinkProtocol

        class InputSchema(PluginSchema):
            value: int

        class MemorySink:
            """Test sink that stores rows in memory."""

            name = "memory"
            input_schema = InputSchema
            idempotent = True
            node_id: str | None = None
            determinism = Determinism.IO_WRITE
            plugin_version = "1.0.0"
            rows: ClassVar[list[dict[str, Any]]] = []

            def __init__(self, config: dict[str, Any]) -> None:
                self.instance_rows: list[dict[str, Any]] = []
                self.config = config

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                self.rows.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="/tmp/memory",
                    content_hash="test",
                    size_bytes=len(str(rows)),
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        sink = MemorySink({})

        # IMPORTANT: Verify protocol conformance at runtime
        assert isinstance(sink, SinkProtocol), "Must conform to SinkProtocol"

        ctx = PluginContext(run_id="test", config={})

        # Batch write
        artifact = sink.write([{"value": 1}, {"value": 2}], ctx)

        assert len(sink.rows) == 2
        assert sink.rows[0] == {"value": 1}
        assert isinstance(artifact, ArtifactDescriptor)
```

**Step 6: Run all sink protocol tests**

Run: `pytest tests/plugins/test_protocols.py::TestSinkProtocol -v`

Expected: All PASS

**Step 7: Commit**

```bash
git add src/elspeth/plugins/protocols.py tests/plugins/test_protocols.py
git commit -m "$(cat <<'EOF'
feat(protocols): update SinkProtocol.write() to batch mode

BREAKING: write(row: dict) -> None is now write(rows: list[dict]) -> ArtifactDescriptor

This matches plugin-protocol contract v1.1 which requires:
- Batch input for efficiency
- ArtifactDescriptor return with content_hash and size_bytes for audit

Actual sink implementations will be updated in WP-03.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update BaseSink.write() signature to batch mode

**Files:**
- Modify: `src/elspeth/plugins/base.py:272-278`
- Test: `tests/plugins/test_base.py`

**Step 1: Write the failing test**

Add test for new batch signature in BaseSink:

```python
# Add to tests/plugins/test_base.py in TestBaseSink class

    def test_base_sink_batch_write_signature(self) -> None:
        """BaseSink.write() accepts batch and returns ArtifactDescriptor."""
        import inspect
        from elspeth.plugins.base import BaseSink

        sig = inspect.signature(BaseSink.write)
        params = list(sig.parameters.keys())

        assert "rows" in params, "write() should accept 'rows' (batch)"
        assert "row" not in params, "write() should NOT have 'row' parameter"

    def test_base_sink_batch_implementation(self) -> None:
        """Test BaseSink subclass with batch write."""
        from typing import Any

        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.plugins.base import BaseSink
        from elspeth.plugins.context import PluginContext

        class InputSchema(PluginSchema):
            value: int

        class BatchMemorySink(BaseSink):
            name = "batch_memory"
            input_schema = InputSchema
            idempotent = True

            def __init__(self, config: dict[str, Any]) -> None:
                super().__init__(config)
                self.rows: list[dict[str, Any]] = []

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                self.rows.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="/tmp/batch",
                    content_hash="hash123",
                    size_bytes=100,
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = BatchMemorySink({})
        ctx = PluginContext(run_id="test", config={})

        artifact = sink.write([{"value": 1}, {"value": 2}, {"value": 3}], ctx)

        assert len(sink.rows) == 3
        assert isinstance(artifact, ArtifactDescriptor)
        assert artifact.content_hash == "hash123"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_base.py::TestBaseSink::test_base_sink_batch_write_signature -v`

Expected: FAIL with `AssertionError: write() should accept 'rows' (batch)`

**Step 3: Update BaseSink.write() signature**

Edit `src/elspeth/plugins/base.py`. First add import at top:

```python
from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
```

Then update BaseSink class (lines 237-300):

```python
class BaseSink(ABC):
    """Base class for sink plugins.

    Subclass and implement write(), flush(), close().

    Example:
        class CSVSink(BaseSink):
            name = "csv"
            input_schema = RowSchema
            idempotent = False

            def write(self, rows: list[dict], ctx: PluginContext) -> ArtifactDescriptor:
                for row in rows:
                    self._writer.writerow(row)
                return ArtifactDescriptor.for_file(
                    path=self._path,
                    content_hash=self._compute_hash(),
                    size_bytes=self._file.tell(),
                )

            def flush(self) -> None:
                self._file.flush()

            def close(self) -> None:
                self._file.close()
    """

    name: str
    input_schema: type[PluginSchema]
    idempotent: bool = False
    node_id: str | None = None  # Set by orchestrator after registration

    # Metadata for Phase 3 audit/reproducibility
    determinism: Determinism = Determinism.IO_WRITE
    plugin_version: str = "0.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config

    @abstractmethod
    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> ArtifactDescriptor:
        """Write a batch of rows to the sink.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash and size_bytes
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush buffered data."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close and release resources."""
        ...

    # === Lifecycle Hooks (Phase 3) ===

    def on_register(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called when plugin is registered."""

    def on_start(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at start of run."""

    def on_complete(self, ctx: PluginContext) -> None:  # noqa: B027
        """Called at end of run (before close)."""
```

**Step 4: Run tests to verify new tests pass**

Run: `pytest tests/plugins/test_base.py::TestBaseSink::test_base_sink_batch_write_signature tests/plugins/test_base.py::TestBaseSink::test_base_sink_batch_implementation -v`

Expected: PASS

**Step 5: Update old test to use new signature**

Update `test_base_sink_implementation` in TestBaseSink:

```python
    def test_base_sink_implementation(self) -> None:
        from typing import Any

        from elspeth.contracts import ArtifactDescriptor, PluginSchema
        from elspeth.plugins.base import BaseSink
        from elspeth.plugins.context import PluginContext

        class InputSchema(PluginSchema):
            value: int

        class MemorySink(BaseSink):
            name = "memory"
            input_schema = InputSchema
            idempotent = True

            def __init__(self, config: dict[str, Any]) -> None:
                super().__init__(config)
                self.rows: list[dict[str, Any]] = []

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                self.rows.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="/tmp/memory",
                    content_hash="test",
                    size_bytes=len(str(rows)),
                )

            def flush(self) -> None:
                pass

            def close(self) -> None:
                pass

        sink = MemorySink({})
        ctx = PluginContext(run_id="test", config={})

        artifact = sink.write([{"value": 1}, {"value": 2}], ctx)

        assert len(sink.rows) == 2
        assert sink.rows[0] == {"value": 1}
        assert isinstance(artifact, ArtifactDescriptor)
```

**Step 6: Run all BaseSink tests**

Run: `pytest tests/plugins/test_base.py::TestBaseSink -v`

Expected: All PASS

**Step 7: Commit**

```bash
git add src/elspeth/plugins/base.py tests/plugins/test_base.py
git commit -m "$(cat <<'EOF'
feat(base): update BaseSink.write() to batch mode

BREAKING: write(row: dict) -> None is now write(rows: list[dict]) -> ArtifactDescriptor

Also changed default determinism from DETERMINISTIC to IO_WRITE since
sinks inherently perform I/O operations.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Verify type checking passes

**Files:**
- All modified files

**Step 1: Run mypy on plugins module**

Run: `mypy src/elspeth/plugins/ --strict`

Expected: May see errors in actual sink implementations (csv_sink.py, etc.) - these are expected until WP-03 fixes them.

**Step 2: Run mypy specifically on protocols and base**

Run: `mypy src/elspeth/plugins/protocols.py src/elspeth/plugins/base.py --strict`

Expected: PASS with no errors on these two files

**Step 3: Run full plugin test suite**

Run: `pytest tests/plugins/ -v --ignore=tests/plugins/sinks/ --ignore=tests/plugins/gates/`

Expected: All PASS (ignoring sink tests which will fail until WP-03)

**Step 4: Document known failures**

Create a note about expected failures:

```bash
echo "WP-01 complete. Expected failures until WP-03:
- src/elspeth/plugins/sinks/csv_sink.py - old write() signature
- src/elspeth/plugins/sinks/json_sink.py - old write() signature
- src/elspeth/plugins/sinks/database_sink.py - old write() signature
- tests/plugins/sinks/* - old test patterns
" > /tmp/wp01-status.txt
```

**Step 5: Commit verification**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: verify WP-01 protocol alignment complete

Verified:
- SourceProtocol has determinism and plugin_version
- BaseSource has determinism and plugin_version with defaults
- SinkProtocol.write() uses batch signature
- BaseSink.write() uses batch signature
- Type checking passes on protocols.py and base.py

Known issues (to be fixed in WP-03):
- Actual sink implementations still use old signature

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist

After completing all tasks:

- [ ] `SourceProtocol` has `determinism: Determinism` attribute
- [ ] `SourceProtocol` has `plugin_version: str` attribute
- [ ] `BaseSource` has `determinism = Determinism.IO_READ` default
- [ ] `BaseSource` has `plugin_version = "0.0.0"` default
- [ ] `SinkProtocol.write()` signature is `write(rows: list[dict], ctx) -> ArtifactDescriptor`
- [ ] `BaseSink.write()` signature is `write(rows: list[dict], ctx) -> ArtifactDescriptor`
- [ ] `BaseSink` has `determinism = Determinism.IO_WRITE` default
- [ ] `mypy src/elspeth/plugins/protocols.py src/elspeth/plugins/base.py --strict` passes
- [ ] `pytest tests/plugins/test_protocols.py tests/plugins/test_base.py -v` passes

---

## Post-Implementation Notes

**What this unlocks:**
- WP-03 can now implement actual sink rewrites knowing the target signature
- Protocol conformance checks will fail for old sinks, providing clear migration guidance

**Expected downstream impacts:**
- `src/elspeth/plugins/sinks/csv_sink.py` - mypy will fail (WP-03)
- `src/elspeth/plugins/sinks/json_sink.py` - mypy will fail (WP-03)
- `src/elspeth/plugins/sinks/database_sink.py` - mypy will fail (WP-03)
- `src/elspeth/engine/adapters.py` - will need update (WP-04)
- `tests/plugins/sinks/*` - will need rewrite (WP-13)
