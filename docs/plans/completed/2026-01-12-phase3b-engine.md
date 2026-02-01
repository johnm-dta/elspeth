# Phase 3B: Engine - SDA Execution (Tasks 11-20)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the SDA Engine executors that wrap plugin calls with audit recording and OpenTelemetry spans.

**Architecture:** SpanFactory provides OpenTelemetry integration. TokenManager handles row instance identity. Executors (TransformExecutor, GateExecutor, etc.) wrap plugin calls. RetryManager handles tenacity-based retries. RowProcessor coordinates single-row processing. Orchestrator manages full run lifecycle.

**Tech Stack:** Python 3.11+, SQLAlchemy Core (database), OpenTelemetry (tracing), tenacity (retries), structlog (logging)

**Dependencies:**
- Phase 1: `elspeth.core.canonical`, `elspeth.core.config`, `elspeth.core.dag`, `elspeth.core.payload_store`, `elspeth.core.landscape.models`
- Phase 2: `elspeth.plugins` (protocols, results, context, schemas, manager, enums, PluginSpec)
- Phase 3A: `elspeth.core.landscape` (schema, db, recorder)

**Phase 2 Additions (use these):**
- `NodeType` enum: Use instead of string literals like `"transform"`
- `Determinism` enum: For reproducibility grading
- `PluginSpec.from_plugin()`: For plugin registration metadata
- Note: `TransformResult.status` is now `"success" | "error"` only (no "route")

**Status Vocabulary (Two Distinct Concepts):**

Phase 3B executors must understand the difference between *processing status* and *terminal state*:

**1. `node_states.status` - Processing status at a single node:**

| Status | Meaning | When Used |
|--------|---------|-----------|
| `open` | Transform is currently executing | `record_node_state()` called |
| `completed` | Transform finished successfully | Transform returned without error |
| `failed` | Transform failed (may be retried) | Transform raised exception |

**2. Token Terminal States - Derived, NOT stored:**

Terminal states are *computed* from relationships, not stored in a column:

| Terminal State | How Derived |
|----------------|-------------|
| `COMPLETED` | Last node_state is at sink node with status=completed |
| `ROUTED` | Has routing_event with move mode to a sink |
| `FORKED` | Token has children in token_parents table |
| `CONSUMED_IN_BATCH` | Token exists in batch_members table |
| `COALESCED` | Token is a parent in token_parents for a join |
| `QUARANTINED` | Last node_state.status=failed + error has quarantine flag |
| `FAILED` | Last node_state.status=failed, no quarantine |

**Other status columns:**
- `runs.status`: `"running"`, `"completed"`, `"failed"`
- `batches.status`: `"draft"`, `"executing"`, `"completed"`, `"failed"`

---

## Critical Audit Invariants

**Every executor MUST satisfy these invariants. Violating them means the audit trail lies.**

For every plugin call:

1. **Node State Exists**: You can point at a `node_states` row and say "this token visited this node at this step; here's input hash, output hash, duration, status."

2. **Routing is Recorded**: If the plugin routed, there are `routing_events` tied to that node_state, and node_state status reflects routed/forked.

3. **Batching is Recorded**: If the plugin batched, there is a `batch` with `batch_members`, and a node_state that marks the token consumed.

4. **Sink Output is Recorded**: If the plugin wrote output, there is a sink node_state *and* an `artifact` record.

5. **Nothing is Inferred Later**: No "pick first non-default sink" logic. Audit trails are recorded facts, not vibes.

**Token Identity Contract:**
- Tokens flow through the entire pipeline, including sinks
- Buffer `TokenInfo`, not dict rows
- Every executor receives token context and records against it

**Status Alignment Contract:**
Each executor sets `node_states.status` based on the *semantic outcome*:

| Plugin Type | Outcome | Status |
|-------------|---------|--------|
| Transform | Success | `completed` |
| Transform | Error | `failed` |
| Gate | continue | `completed` |
| Gate | route_to_sink | `completed` (routing_event records the route) |
| Gate | fork_to_paths | `completed` (child tokens created) |
| Aggregation | accept | `completed` (batch_members records consumption) |
| Aggregation | flush | `completed` |
| Sink | write | `completed` |

**Note:** Terminal states (ROUTED, FORKED, CONSUMED_IN_BATCH) are *derived* from relationships—they're not stored in `node_states.status`. A gate that routes still has `status=completed`; the routing_event proves the route.

---

## Task 11: SpanFactory - OpenTelemetry Integration

**Files:**
- Create: `src/elspeth/engine/spans.py`
- Create: `tests/engine/test_spans.py`

### Step 1: Write the failing test

```python
# tests/engine/test_spans.py
"""Tests for OpenTelemetry span factory."""

import pytest


class TestSpanFactory:
    """OpenTelemetry span creation."""

    def test_create_run_span(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()  # No tracer = no-op mode

        with factory.run_span("run-001") as span:
            # No-op mode returns NoOpSpan (not None) so callers can use uniform interface
            assert isinstance(span, NoOpSpan)

    def test_create_row_span(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()

        with factory.row_span("row-001", "token-001") as span:
            assert isinstance(span, NoOpSpan)

    def test_create_transform_span(self) -> None:
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        factory = SpanFactory()

        with factory.transform_span("my_transform", input_hash="abc123") as span:
            assert isinstance(span, NoOpSpan)

    def test_with_tracer(self) -> None:
        """Test with actual tracer if opentelemetry available."""
        pytest.importorskip("opentelemetry")
        from opentelemetry.sdk.trace import TracerProvider

        from elspeth.engine.spans import SpanFactory

        # Create provider locally - do NOT set global state with set_tracer_provider()
        # Global state causes flaky tests when other tests/libraries have already set a provider
        provider = TracerProvider()
        tracer = provider.get_tracer("test")  # Get tracer from local provider

        factory = SpanFactory(tracer=tracer)

        with factory.run_span("run-001") as span:
            assert span is not None
            assert span.is_recording()

    def test_noop_span_interface(self) -> None:
        """Test that NoOpSpan has the same interface as real spans."""
        from elspeth.engine.spans import NoOpSpan, SpanFactory

        # NoOpSpan should be usable in place of real spans
        noop = NoOpSpan()
        noop.set_attribute("key", "value")  # Should not raise
        noop.set_status(None)  # Should not raise
        noop.record_exception(ValueError("test"))  # Should not raise
        assert noop.is_recording() is False

        # Factory in no-op mode should return NoOpSpan, not None
        factory = SpanFactory()  # No tracer = no-op mode
        with factory.run_span("run-001") as span:
            assert isinstance(span, NoOpSpan)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_spans.py -v`
Expected: FAIL (ImportError)

### Step 3: Create spans module

```python
# src/elspeth/engine/spans.py
"""OpenTelemetry span factory for SDA Engine.

Provides structured span creation for pipeline execution.
Falls back to no-op mode when no tracer is configured.

Span Hierarchy:
    run:{run_id}
    ├── source:{source_name}
    │   └── load
    ├── row:{row_id}
    │   ├── transform:{transform_name}
    │   ├── gate:{gate_name}
    │   └── sink:{sink_name}
    └── aggregation:{agg_name}
        └── flush
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op."""
        pass

    def set_status(self, status: Any) -> None:
        """No-op."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op."""
        pass

    def is_recording(self) -> bool:
        """Always False for no-op."""
        return False


class SpanFactory:
    """Factory for creating OpenTelemetry spans.

    When no tracer is provided, all span methods return no-op contexts.

    Example:
        factory = SpanFactory(tracer=opentelemetry.trace.get_tracer("elspeth"))

        with factory.run_span("run-001") as span:
            with factory.row_span("row-001", "token-001") as row_span:
                with factory.transform_span("my_transform") as transform_span:
                    # Do work
                    pass
    """

    def __init__(self, tracer: "Tracer | None" = None) -> None:
        """Initialize with optional tracer.

        Args:
            tracer: OpenTelemetry tracer. If None, spans are no-ops.
        """
        self._tracer = tracer

    @property
    def enabled(self) -> bool:
        """Whether tracing is enabled."""
        return self._tracer is not None

    # Singleton no-op span to avoid repeated allocations
    _NOOP_SPAN = NoOpSpan()

    @contextmanager
    def run_span(self, run_id: str) -> Iterator["Span | NoOpSpan"]:
        """Create a span for the entire run.

        Args:
            run_id: Run identifier

        Yields:
            Span or NoOpSpan if tracing disabled (never None - uniform interface)
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"run:{run_id}") as span:
            span.set_attribute("run.id", run_id)
            yield span

    @contextmanager
    def source_span(self, source_name: str) -> Iterator["Span | NoOpSpan"]:
        """Create a span for source loading.

        Args:
            source_name: Name of the source plugin

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"source:{source_name}") as span:
            span.set_attribute("plugin.name", source_name)
            span.set_attribute("plugin.type", "source")
            yield span

    @contextmanager
    def row_span(
        self,
        row_id: str,
        token_id: str,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for processing a row.

        Args:
            row_id: Row identifier
            token_id: Token identifier

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"row:{row_id}") as span:
            span.set_attribute("row.id", row_id)
            span.set_attribute("token.id", token_id)
            yield span

    @contextmanager
    def transform_span(
        self,
        transform_name: str,
        *,
        input_hash: str | None = None,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for a transform operation.

        Args:
            transform_name: Name of the transform plugin
            input_hash: Optional input data hash

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"transform:{transform_name}") as span:
            span.set_attribute("plugin.name", transform_name)
            span.set_attribute("plugin.type", "transform")
            if input_hash:
                span.set_attribute("input.hash", input_hash)
            yield span

    @contextmanager
    def gate_span(
        self,
        gate_name: str,
        *,
        input_hash: str | None = None,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for a gate operation.

        Args:
            gate_name: Name of the gate plugin
            input_hash: Optional input data hash

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"gate:{gate_name}") as span:
            span.set_attribute("plugin.name", gate_name)
            span.set_attribute("plugin.type", "gate")
            if input_hash:
                span.set_attribute("input.hash", input_hash)
            yield span

    @contextmanager
    def aggregation_span(
        self,
        aggregation_name: str,
        *,
        batch_id: str | None = None,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for an aggregation flush.

        Args:
            aggregation_name: Name of the aggregation plugin
            batch_id: Optional batch identifier

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"aggregation:{aggregation_name}") as span:
            span.set_attribute("plugin.name", aggregation_name)
            span.set_attribute("plugin.type", "aggregation")
            if batch_id:
                span.set_attribute("batch.id", batch_id)
            yield span

    @contextmanager
    def sink_span(
        self,
        sink_name: str,
    ) -> Iterator["Span | NoOpSpan"]:
        """Create a span for a sink write.

        Args:
            sink_name: Name of the sink plugin

        Yields:
            Span or NoOpSpan
        """
        if self._tracer is None:
            yield self._NOOP_SPAN
            return

        with self._tracer.start_as_current_span(f"sink:{sink_name}") as span:
            span.set_attribute("plugin.name", sink_name)
            span.set_attribute("plugin.type", "sink")
            yield span
```

### Step 4: Create engine directory

```bash
mkdir -p src/elspeth/engine tests/engine
touch src/elspeth/engine/__init__.py tests/engine/__init__.py
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/engine/test_spans.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/elspeth/engine/ tests/engine/
git commit -m "feat(engine): add SpanFactory for OpenTelemetry integration"
```

---

## Task 12: TokenManager - High-Level Token Operations

**Files:**
- Create: `src/elspeth/engine/tokens.py`
- Create: `tests/engine/test_tokens.py`

### Step 1: Write the failing test

```python
# tests/engine/test_tokens.py
"""Tests for TokenManager."""

import pytest


class TestTokenManager:
    """High-level token management."""

    def test_create_initial_token(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        manager = TokenManager(recorder)

        token_info = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        assert token_info.row_id is not None
        assert token_info.token_id is not None
        assert token_info.row_data == {"value": 42}

    def test_fork_token(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        manager = TokenManager(recorder)
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        # step_in_pipeline is required - Orchestrator/RowProcessor is the authority
        children = manager.fork_token(
            parent_token=initial,
            branches=["stats", "classifier"],
            step_in_pipeline=1,  # Fork happens at step 1
        )

        assert len(children) == 2
        assert children[0].branch_name == "stats"
        assert children[1].branch_name == "classifier"
        # Children inherit row_data
        assert children[0].row_data == {"value": 42}

    def test_update_row_data(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        manager = TokenManager(recorder)
        token_info = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"x": 1},
        )

        updated = manager.update_row_data(
            token_info,
            new_data={"x": 1, "y": 2},
        )

        assert updated.row_data == {"x": 1, "y": 2}
        assert updated.token_id == token_info.token_id  # Same token
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_tokens.py -v`
Expected: FAIL (ImportError)

### Step 3: Create tokens module

```python
# src/elspeth/engine/tokens.py
"""TokenManager: High-level token operations for the SDA engine.

Provides a simplified interface over LandscapeRecorder for managing
tokens (row instances flowing through the DAG).
"""

from dataclasses import dataclass, field
from typing import Any

from elspeth.core.landscape import LandscapeRecorder


@dataclass
class TokenInfo:
    """Information about a token in flight.

    Carries both identity (IDs) and current state (row_data).

    Note: Step position is NOT tracked here - the Orchestrator/RowProcessor
    is the authority for where a token is in the DAG. TokenInfo is just
    identity + payload, not position.
    """

    row_id: str
    token_id: str
    row_data: dict[str, Any]
    branch_name: str | None = None


class TokenManager:
    """Manages token lifecycle for the SDA engine.

    Provides high-level operations:
    - Create initial token from source row
    - Fork token to multiple branches
    - Coalesce tokens from branches
    - Update token row data after transforms

    Example:
        manager = TokenManager(recorder)

        # Create token for source row
        token = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        # After transform
        token = manager.update_row_data(token, {"value": 42, "processed": True})

        # Fork to branches (step_in_pipeline from Orchestrator)
        children = manager.fork_token(
            parent_token=token,
            branches=["stats", "classifier"],
            step_in_pipeline=2,  # Orchestrator provides step position
        )
    """

    def __init__(self, recorder: LandscapeRecorder) -> None:
        """Initialize with recorder.

        Args:
            recorder: LandscapeRecorder for audit trail
        """
        self._recorder = recorder

    def create_initial_token(
        self,
        run_id: str,
        source_node_id: str,
        row_index: int,
        row_data: dict[str, Any],
    ) -> TokenInfo:
        """Create a token for a source row.

        Args:
            run_id: Run identifier
            source_node_id: Source node that loaded the row
            row_index: Position in source (0-indexed)
            row_data: Row data from source

        Returns:
            TokenInfo with row and token IDs
        """
        # Create row record
        row = self._recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            data=row_data,
        )

        # Create initial token
        token = self._recorder.create_token(row_id=row.row_id)

        return TokenInfo(
            row_id=row.row_id,
            token_id=token.token_id,
            row_data=row_data,
        )

    def fork_token(
        self,
        parent_token: TokenInfo,
        branches: list[str],
        step_in_pipeline: int,
        row_data: dict[str, Any] | None = None,
    ) -> list[TokenInfo]:
        """Fork a token to multiple branches.

        The step_in_pipeline is required because the Orchestrator/RowProcessor
        owns step position - TokenManager doesn't track it.

        Args:
            parent_token: Parent token to fork
            branches: List of branch names
            step_in_pipeline: Current step position in the DAG
            row_data: Optional row data (defaults to parent's data)

        Returns:
            List of child TokenInfo, one per branch
        """
        data = row_data if row_data is not None else parent_token.row_data

        children = self._recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=parent_token.row_id,
            branches=branches,
            step_in_pipeline=step_in_pipeline,
        )

        return [
            TokenInfo(
                row_id=parent_token.row_id,
                token_id=child.token_id,
                row_data=data.copy(),
                branch_name=child.branch_name,
            )
            for child in children
        ]

    def coalesce_tokens(
        self,
        parents: list[TokenInfo],
        merged_data: dict[str, Any],
        step_in_pipeline: int,
    ) -> TokenInfo:
        """Coalesce multiple tokens into one.

        The step_in_pipeline is required because the Orchestrator/RowProcessor
        owns step position - TokenManager doesn't track it.

        Args:
            parents: Parent tokens to merge
            merged_data: Merged row data
            step_in_pipeline: Current step position in the DAG

        Returns:
            Merged TokenInfo
        """
        # Use first parent's row_id (they should all be the same)
        row_id = parents[0].row_id

        merged = self._recorder.coalesce_tokens(
            parent_token_ids=[p.token_id for p in parents],
            row_id=row_id,
            step_in_pipeline=step_in_pipeline,
        )

        return TokenInfo(
            row_id=row_id,
            token_id=merged.token_id,
            row_data=merged_data,
        )

    def update_row_data(
        self,
        token: TokenInfo,
        new_data: dict[str, Any],
    ) -> TokenInfo:
        """Update token's row data after a transform.

        Args:
            token: Token to update
            new_data: New row data

        Returns:
            Updated TokenInfo (same token_id, new row_data)
        """
        return TokenInfo(
            row_id=token.row_id,
            token_id=token.token_id,
            row_data=new_data,
            branch_name=token.branch_name,
        )

    # NOTE: No advance_step() method - step position is the authority of
    # Orchestrator/RowProcessor, not TokenManager. They track where tokens
    # are in the DAG and pass step_in_pipeline when needed.
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_tokens.py -v`
Expected: PASS

### Step 5: Phase 3A Dependency

**IMPORTANT**: This task requires Phase 3A's `LandscapeRecorder.fork_token()` and
`LandscapeRecorder.coalesce_tokens()` to accept a `step_in_pipeline` parameter:

```python
# In LandscapeRecorder.fork_token():
def fork_token(
    self,
    parent_token_id: str,
    row_id: str,
    branches: list[str],
    step_in_pipeline: int,  # NEW PARAMETER
) -> list[Token]:
    # ... insert into token_parents with step_in_pipeline
```

The `token_parents` table has a `step_in_pipeline` column that tracks when the
fork/join occurred in the pipeline. Without this, we'd lose DAG position audit info.

### Step 6: Commit

```bash
git add src/elspeth/engine/tokens.py tests/engine/test_tokens.py
git commit -m "feat(engine): add TokenManager for high-level token operations"
```

---

## Task 13: TransformExecutor - Audit-Wrapped Transform Execution

**Files:**
- Create: `src/elspeth/engine/executors.py`
- Create: `tests/engine/test_executors.py`

### Step 1: Write the failing test

```python
# tests/engine/test_executors.py
"""Tests for plugin executors."""

import pytest


class TestTransformExecutor:
    """Transform execution with audit."""

    def test_execute_transform_success(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo, TokenManager
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="double",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        # Mock transform plugin
        class DoubleTransform:
            name = "double"
            node_id = node.node_id

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({"value": row["value"] * 2})

        transform = DoubleTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 21},
        )

        # Need to create row/token in landscape first
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, updated_token = executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,  # First transform is at step 1 (source=0)
        )

        assert result.status == "success"
        assert result.row == {"value": 42}
        # Audit fields populated
        assert result.input_hash is not None
        assert result.output_hash is not None
        assert result.duration_ms is not None

    def test_execute_transform_error(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="failing",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        class FailingTransform:
            name = "failing"
            node_id = node.node_id

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.error({"message": "validation failed"})

        transform = FailingTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": -1},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, _ = executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,  # First transform is at step 1 (source=0)
        )

        assert result.status == "error"
        assert result.reason == {"message": "validation failed"}
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_executors.py::TestTransformExecutor -v`
Expected: FAIL (ImportError)

### Step 3: Create executors module

```python
# src/elspeth/engine/executors.py
"""Plugin executors that wrap plugin calls with audit recording.

Each executor handles a specific plugin type:
- TransformExecutor: Row transforms
- GateExecutor: Routing gates
- AggregationExecutor: Stateful aggregations
- SinkExecutor: Output sinks
"""

import time
from dataclasses import dataclass
from typing import Any, Protocol

from elspeth.core.canonical import stable_hash
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenInfo
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult, TransformResult


class TransformLike(Protocol):
    """Protocol for transform-like plugins."""

    name: str
    node_id: str

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process a row."""
        ...


class TransformExecutor:
    """Executes transforms with audit recording.

    Wraps transform.process() to:
    1. Record node state start
    2. Time the operation
    3. Populate audit fields in result
    4. Record node state completion
    5. Emit OpenTelemetry span

    Example:
        executor = TransformExecutor(recorder, span_factory)
        result, updated_token = executor.execute_transform(
            transform=my_transform,
            token=token,
            ctx=ctx,
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
        """
        self._recorder = recorder
        self._spans = span_factory

    def execute_transform(
        self,
        transform: TransformLike,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
    ) -> tuple[TransformResult, TokenInfo]:
        """Execute a transform with full audit recording.

        This method handles a SINGLE ATTEMPT. Retry logic is the caller's
        responsibility (e.g., RetryManager wraps this for retryable transforms).
        Each attempt gets its own node_state record with attempt number tracked
        by the caller.

        Args:
            transform: Transform plugin to execute
            token: Current token with row data
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)

        Returns:
            Tuple of (TransformResult with audit fields, updated TokenInfo)

        Raises:
            Exception: Re-raised from transform.process() after recording failure
        """
        input_hash = stable_hash(token.row_data)

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=transform.node_id,
            step_index=step_in_pipeline,
            input_data=token.row_data,
        )

        # Execute with timing and span
        with self._spans.transform_span(transform.name, input_hash=input_hash):
            start = time.perf_counter()
            try:
                result = transform.process(token.row_data, ctx)
                duration_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                # Record failure
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status="failed",
                    duration_ms=duration_ms,
                    error={"exception": str(e), "type": type(e).__name__},
                )
                raise

        # Populate audit fields
        result.input_hash = input_hash
        result.output_hash = stable_hash(result.row) if result.row else None
        result.duration_ms = duration_ms

        # Complete node state
        if result.status == "success":
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="completed",
                output_data=result.row,
                duration_ms=duration_ms,
            )
            # Update token with new row data
            updated_token = TokenInfo(
                row_id=token.row_id,
                token_id=token.token_id,
                row_data=result.row,
                branch_name=token.branch_name,
            )
        else:
            # Transform returned error status (not exception)
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="failed",
                duration_ms=duration_ms,
                error=result.reason,
            )
            updated_token = token

        return result, updated_token
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_executors.py::TestTransformExecutor -v`
Expected: PASS

### Step 5: RetryManager Integration Note

**How TransformExecutor integrates with RetryManager:**

TransformExecutor handles a *single attempt*. Retry logic is external:

```python
# In RowProcessor/Orchestrator (not TransformExecutor):
def execute_with_retry(transform, token, ctx, step, retry_manager):
    """Execute transform with retry wrapping."""
    attempt = 0

    @retry_manager.with_retry(transform.retry_policy)
    def _attempt():
        nonlocal attempt
        attempt += 1
        # Each attempt gets its own node_state
        return executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=step,
            # Note: attempt number is tracked in node_state via Landscape
        )

    return _attempt()
```

This separation keeps TransformExecutor focused on audit recording while RetryManager
handles backoff, limits, and exception classification. See Task 17 for RetryManager.

### Step 6: Commit

```bash
git add src/elspeth/engine/executors.py tests/engine/test_executors.py
git commit -m "feat(engine): add TransformExecutor with audit recording"
```

---

## Task 14: GateExecutor - Routing with Audit

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Modify: `tests/engine/test_executors.py`

### Step 1: Write the failing tests

```python
# Add to tests/engine/test_executors.py

class TestGateExecutor:
    """Gate execution with routing audit."""

    def test_execute_gate_continue(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="threshold",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )

        class ThresholdGate:
            name = "threshold"
            node_id = gate_node.node_id

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                if row["value"] < 100:
                    return GateResult(row=row, action=RoutingAction.continue_())
                return GateResult(
                    row=row,
                    action=RoutingAction.route_to_sink("high_values"),
                )

        gate = ThresholdGate()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 50},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        outcome = executor.execute_gate(
            gate=gate, token=token, ctx=ctx, step_in_pipeline=0
        )

        assert outcome.result.action.kind == "continue"

    def test_execute_gate_route(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="threshold",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="high_values",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=sink_node.node_id,
            label="high_values",
            mode="move",
        )

        class ThresholdGate:
            name = "threshold"
            node_id = gate_node.node_id

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.route_to_sink(
                        "high_values",
                        reason={"value": row["value"]},
                    ),
                )

        gate = ThresholdGate()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory(), edge_map={
            (gate_node.node_id, "high_values"): edge.edge_id,
        })

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 200},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        outcome = executor.execute_gate(
            gate=gate, token=token, ctx=ctx, step_in_pipeline=0
        )

        assert outcome.result.action.kind == "route_to_sink"
        assert outcome.result.action.destinations == ("high_values",)
        assert outcome.sink_name == "high_values"  # GateOutcome captures destination

    def test_missing_edge_raises_error(self) -> None:
        """Missing edge registration must raise, not silently drop routing."""
        import pytest
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor, MissingEdgeError
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="threshold",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )
        # NOTE: No sink node registered, no edge registered

        class ThresholdGate:
            name = "threshold"
            node_id = gate_node.node_id

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.route_to_sink("nonexistent_sink"),
                )

        gate = ThresholdGate()
        ctx = PluginContext(run_id=run.run_id, config={})
        # Empty edge_map - no edges registered
        executor = GateExecutor(recorder, SpanFactory(), edge_map={})

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 200},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        with pytest.raises(MissingEdgeError) as exc_info:
            executor.execute_gate(
                gate=gate, token=token, ctx=ctx, step_in_pipeline=0
            )

        assert exc_info.value.node_id == gate_node.node_id
        assert exc_info.value.label == "nonexistent_sink"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_executors.py::TestGateExecutor -v`
Expected: FAIL

### Step 3: Add GateExecutor

```python
# Add to src/elspeth/engine/executors.py

from elspeth.plugins.results import GateResult, RoutingAction


class MissingEdgeError(Exception):
    """Raised when routing refers to an unregistered edge.

    This is an audit integrity error - silent edge loss is unacceptable.
    """

    def __init__(self, node_id: str, label: str) -> None:
        self.node_id = node_id
        self.label = label
        super().__init__(
            f"No edge registered from node {node_id} with label '{label}'. "
            "Audit trail would be incomplete - refusing to proceed."
        )


@dataclass
class GateOutcome:
    """Result of gate execution with routing information.

    This structured return ensures routing destinations are explicit,
    not inferred later by the orchestrator.
    """

    result: GateResult
    updated_token: TokenInfo
    child_tokens: list[TokenInfo]  # Non-empty only for fork_to_paths
    sink_name: str | None  # Non-None only for route_to_sink


class GateLike(Protocol):
    """Protocol for gate-like plugins."""

    name: str
    node_id: str

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate routing decision."""
        ...


class GateExecutor:
    """Executes gates with routing audit recording.

    Wraps gate.evaluate() to:
    1. Record node state start
    2. Time the operation
    3. Record routing events
    4. Record node state completion
    5. Emit OpenTelemetry span

    Example:
        executor = GateExecutor(recorder, span_factory, edge_map)
        result, updated_token = executor.execute_gate(
            gate=my_gate,
            token=token,
            ctx=ctx,
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        edge_map: dict[tuple[str, str], str] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            edge_map: Map of (node_id, label) -> edge_id
        """
        self._recorder = recorder
        self._spans = span_factory
        self._edge_map = edge_map or {}

    def execute_gate(
        self,
        gate: GateLike,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
        token_manager: "TokenManager | None" = None,
    ) -> "GateOutcome":
        """Execute a gate with full audit recording.

        Args:
            gate: Gate plugin to execute
            token: Current token with row data
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)
            token_manager: Required for fork_to_paths to create child tokens

        Returns:
            GateOutcome containing result, updated token(s), and routing info

        Raises:
            MissingEdgeError: If routing refers to an unregistered edge (audit loss prevention)
        """
        input_hash = stable_hash(token.row_data)

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            step_index=step_in_pipeline,  # Orchestrator is authority for position
            input_data=token.row_data,
        )

        # Execute with timing and span
        with self._spans.gate_span(gate.name, input_hash=input_hash):
            start = time.perf_counter()
            try:
                result = gate.evaluate(token.row_data, ctx)
                duration_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status="failed",
                    duration_ms=duration_ms,
                    error={"exception": str(e), "type": type(e).__name__},
                )
                raise

        # Populate audit fields
        result.input_hash = input_hash
        result.output_hash = stable_hash(result.row)
        result.duration_ms = duration_ms

        # Handle routing based on action kind
        child_tokens: list[TokenInfo] = []
        sink_name: str | None = None

        if result.action.kind == "continue":
            # Simple continue - single updated token
            updated_token = TokenInfo(
                row_id=token.row_id,
                token_id=token.token_id,
                row_data=result.row,
                branch_name=token.branch_name,
            )
        elif result.action.kind == "route_to_sink":
            # Route to sink - record routing event, return sink destination
            sink_name = result.action.destinations[0]
            self._record_routing(state.state_id, gate.node_id, result.action)
            updated_token = TokenInfo(
                row_id=token.row_id,
                token_id=token.token_id,
                row_data=result.row,
                branch_name=token.branch_name,
            )
        elif result.action.kind == "fork_to_paths":
            # Fork - create child tokens for each path
            if token_manager is None:
                raise ValueError("token_manager required for fork_to_paths")

            self._record_routing(state.state_id, gate.node_id, result.action)
            child_tokens = token_manager.fork_token(
                parent_token=token,
                branches=list(result.action.destinations),
                step_in_pipeline=step_in_pipeline,
                row_data=result.row,
            )
            # Parent token terminates here - return first child as "updated"
            updated_token = child_tokens[0] if child_tokens else token
        else:
            raise ValueError(f"Unknown action kind: {result.action.kind}")

        # Complete node state - always "completed" for successful execution
        # Terminal state (ROUTED, FORKED) is derived from routing_events/token_parents
        self._recorder.complete_node_state(
            state_id=state.state_id,
            status="completed",
            output_data=result.row,
            duration_ms=duration_ms,
        )

        return GateOutcome(
            result=result,
            updated_token=updated_token,
            child_tokens=child_tokens,
            sink_name=sink_name,
        )

    def _record_routing(
        self,
        state_id: str,
        node_id: str,
        action: RoutingAction,
    ) -> None:
        """Record routing events for a routing action.

        Raises:
            MissingEdgeError: If any destination has no registered edge.
                This is a hard error - silent audit loss is unacceptable.
        """
        if len(action.destinations) == 1:
            # Single destination - must have registered edge
            dest = action.destinations[0]
            edge_id = self._edge_map.get((node_id, dest))
            if edge_id is None:
                raise MissingEdgeError(node_id=node_id, label=dest)

            self._recorder.record_routing_event(
                state_id=state_id,
                edge_id=edge_id,
                mode=action.mode,
                reason=action.reason,
            )
        else:
            # Multiple destinations (fork) - all must have registered edges
            routes = []
            for dest in action.destinations:
                edge_id = self._edge_map.get((node_id, dest))
                if edge_id is None:
                    raise MissingEdgeError(node_id=node_id, label=dest)
                routes.append({"edge_id": edge_id, "mode": action.mode})

            self._recorder.record_routing_events(
                state_id=state_id,
                routes=routes,
                reason=action.reason,
            )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_executors.py::TestGateExecutor -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(engine): add GateExecutor with routing audit"
```

---

## Task 15: AggregationExecutor - Batch Tracking

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Modify: `tests/engine/test_executors.py`

### Step 1: Write the failing tests

```python
# Add to tests/engine/test_executors.py

class TestAggregationExecutor:
    """Aggregation execution with batch tracking."""

    def test_accept_creates_batch(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        class SumAggregation:
            name = "sum"
            node_id = agg_node.node_id
            _batch_id = None
            _count = 0

            def accept(self, row: dict, ctx: PluginContext) -> AcceptResult:
                self._count += 1
                return AcceptResult(accepted=True, trigger=self._count >= 2)

            def flush(self, ctx: PluginContext) -> list[dict]:
                return [{"sum": 100}]

        agg = SumAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 50},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result = executor.accept(
            aggregation=agg, token=token, ctx=ctx, step_in_pipeline=0
        )

        assert result.accepted is True
        assert result.batch_id is not None  # Batch created

    def test_flush_with_audit(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        class SumAggregation:
            name = "sum"
            node_id = agg_node.node_id
            _batch_id = None
            _buffer = []

            def accept(self, row: dict, ctx: PluginContext) -> AcceptResult:
                self._buffer.append(row["value"])
                return AcceptResult(accepted=True, trigger=len(self._buffer) >= 2)

            def flush(self, ctx: PluginContext) -> list[dict]:
                result = [{"sum": sum(self._buffer)}]
                self._buffer = []
                return result

        agg = SumAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

        # Accept two rows
        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": 50},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            result = executor.accept(
                aggregation=agg, token=token, ctx=ctx, step_in_pipeline=0
            )

        # Flush
        outputs = executor.flush(
            aggregation=agg, ctx=ctx, trigger_reason="threshold", step_in_pipeline=0
        )

        assert len(outputs) == 1
        assert outputs[0] == {"sum": 100}

        # Batch should be completed
        batch = recorder.get_batch(result.batch_id)
        assert batch.status == "completed"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_executors.py::TestAggregationExecutor -v`
Expected: FAIL

### Step 3: Add AggregationExecutor

```python
# Add to src/elspeth/engine/executors.py

from elspeth.plugins.results import AcceptResult


class AggregationLike(Protocol):
    """Protocol for aggregation-like plugins."""

    name: str
    node_id: str
    _batch_id: str | None

    def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
        """Accept a row into the aggregation."""
        ...

    def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
        """Flush the aggregation."""
        ...


class AggregationExecutor:
    """Executes aggregations with batch tracking.

    Manages the batch lifecycle:
    1. Create draft batch on first accept()
    2. Persist batch members immediately (crash-safe)
    3. Transition to executing on flush()
    4. Transition to completed/failed after flush()

    Example:
        executor = AggregationExecutor(recorder, span_factory, run_id)

        # Accept rows
        result = executor.accept(aggregation, token, ctx, step)
        if result.trigger:
            outputs = executor.flush(aggregation, ctx, "threshold", step)
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            run_id: Current run ID
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._member_counts: dict[str, int] = {}  # batch_id -> count

    def accept(
        self,
        aggregation: AggregationLike,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
    ) -> AcceptResult:
        """Accept a row into an aggregation with batch tracking.

        Each accept creates a node_state (open→completed) for audit trail.
        Token's terminal state becomes CONSUMED_IN_BATCH (derived from batch_members).

        Args:
            aggregation: Aggregation plugin
            token: Current token
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)

        Returns:
            AcceptResult with batch_id populated
        """
        # Create batch on first accept
        if aggregation._batch_id is None:
            batch = self._recorder.create_batch(
                run_id=self._run_id,
                aggregation_node_id=aggregation.node_id,
            )
            aggregation._batch_id = batch.batch_id
            self._member_counts[batch.batch_id] = 0

        # Begin node state for this accept operation
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=aggregation.node_id,
            step_index=step_in_pipeline,
            input_data=token.row_data,
        )

        start = time.perf_counter()
        try:
            # Call plugin accept
            result = aggregation.accept(token.row_data, ctx)
            duration_ms = (time.perf_counter() - start) * 1000

            # Only record batch membership if accepted
            # CRITICAL: accepted=False means token was evaluated but rejected
            # by the aggregation's criteria - it should NOT be in the batch
            if result.accepted:
                ordinal = self._member_counts[aggregation._batch_id]
                self._recorder.add_batch_member(
                    batch_id=aggregation._batch_id,
                    token_id=token.token_id,
                    ordinal=ordinal,
                )
                self._member_counts[aggregation._batch_id] = ordinal + 1
                result.batch_id = aggregation._batch_id

            # Complete node state - status reflects acceptance
            # Terminal state (CONSUMED_IN_BATCH) is derived from batch_members table
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="completed" if result.accepted else "rejected",
                duration_ms=duration_ms,
            )

            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="failed",
                duration_ms=duration_ms,
                error={"exception": str(e), "type": type(e).__name__},
            )
            raise

    def flush(
        self,
        aggregation: AggregationLike,
        ctx: PluginContext,
        trigger_reason: str,
        step_in_pipeline: int,
    ) -> list[dict[str, Any]]:
        """Flush an aggregation with status management.

        NOTE: flush() is a batch-level operation, not token-level. The audit
        trail is maintained via batch status transitions rather than node_states.
        The batch_events table (or batch status history) records the flush.

        Args:
            aggregation: Aggregation plugin
            ctx: Plugin context
            trigger_reason: Why flush was triggered
            step_in_pipeline: Current position for output tokens

        Returns:
            List of output rows
        """
        batch_id = aggregation._batch_id
        if batch_id is None:
            return []

        # Transition to executing - this IS the audit record for flush start
        self._recorder.update_batch_status(
            batch_id,
            "executing",
            trigger_reason=trigger_reason,
        )

        start = time.perf_counter()
        with self._spans.aggregation_span(aggregation.name, batch_id=batch_id):
            try:
                outputs = aggregation.flush(ctx)

                # Transition to completed - batch status tracks state machine,
                # not timing. Timing is captured in the aggregation_state_id's
                # node_state (set via state_id parameter when flush creates one).
                self._recorder.update_batch_status(
                    batch_id,
                    "completed",
                )

                # Reset for next batch
                aggregation._batch_id = None
                if batch_id in self._member_counts:
                    del self._member_counts[batch_id]

                return outputs

            except Exception as e:
                # Record failure - error details go in the node_state, not batch
                self._recorder.update_batch_status(
                    batch_id,
                    "failed",
                )
                raise
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_executors.py::TestAggregationExecutor -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(engine): add AggregationExecutor with batch tracking"
```

---

## Task 16: SinkExecutor - Artifact Recording

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Modify: `tests/engine/test_executors.py`

### Step 1: Write the failing tests

```python
# Add to tests/engine/test_executors.py

class TestSinkExecutor:
    """Sink execution with artifact recording."""

    def test_write_records_artifact(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )

        class CSVSink:
            name = "csv_sink"
            node_id = sink_node.node_id

            def write(self, rows: list[dict], ctx: PluginContext) -> dict:
                # Return artifact info
                return {
                    "path": "/output/result.csv",
                    "size_bytes": 1024,
                    "content_hash": "abc123",
                }

        sink = CSVSink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=sink_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        artifact = executor.write(
            sink=sink,
            tokens=[token],
            ctx=ctx,
            step_in_pipeline=1,  # First transform is at step 1 (source=0)
        )

        assert artifact is not None
        assert artifact.path_or_uri == "/output/result.csv"

        # Artifact recorded in Landscape
        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 1
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_executors.py::TestSinkExecutor -v`
Expected: FAIL

### Step 3: Add SinkExecutor

```python
# Add to src/elspeth/engine/executors.py

from elspeth.core.landscape import Artifact


class SinkLike(Protocol):
    """Engine-internal protocol for sink execution.

    NOTE: This is NOT the same as SinkProtocol from Phase 2.
    SinkProtocol.write() takes a single row and returns None.

    SinkLike is an adapter interface - the SinkAdapter class (Task 8.5
    in Phase 4) bridges between real SinkProtocol plugins and this
    interface by:
    1. Looping over rows to call sink.write(row, ctx) individually
    2. Calling sink.flush() after all rows
    3. Computing and returning artifact metadata (hash, path, etc.)

    This separation allows the engine to think in terms of batches
    while plugins remain simple single-row writers.
    """

    name: str
    node_id: str

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> dict[str, Any]:
        """Write rows and return artifact info (hash, path, size)."""
        ...


class SinkExecutor:
    """Executes sinks with artifact recording.

    Wraps sink.write() to:
    1. Record node state for each input token
    2. Time the operation
    3. Register resulting artifact
    4. Emit OpenTelemetry span

    Example:
        executor = SinkExecutor(recorder, span_factory, run_id)
        artifact = executor.write(sink, tokens, ctx)
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            run_id: Current run ID
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id

    def write(
        self,
        sink: SinkLike,
        tokens: list[TokenInfo],
        ctx: PluginContext,
        step_in_pipeline: int,
    ) -> Artifact | None:
        """Write tokens to sink with artifact recording.

        CRITICAL: Creates a node_state for EACH token written. This is how
        we derive the COMPLETED terminal state - every token that reaches
        a sink gets a completed node_state at the sink node.

        Args:
            sink: Sink plugin
            tokens: Tokens to write (preserves TokenInfo, not just dicts!)
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)

        Returns:
            Artifact if produced, None otherwise
        """
        if not tokens:
            return None

        rows = [t.row_data for t in tokens]

        # Create node_state for EACH token - this is how we know they reached the sink
        # and can derive COMPLETED terminal state
        states: list[tuple[TokenInfo, Any]] = []  # (token, state) pairs
        for token in tokens:
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=sink.node_id,
                step_index=step_in_pipeline,
                input_data=token.row_data,
            )
            states.append((token, state))

        with self._spans.sink_span(sink.name):
            start = time.perf_counter()
            try:
                artifact_info = sink.write(rows, ctx)
                duration_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                # Mark all token states as failed
                for _, state in states:
                    self._recorder.complete_node_state(
                        state_id=state.state_id,
                        status="failed",
                        duration_ms=duration_ms,
                        error={"exception": str(e), "type": type(e).__name__},
                    )
                raise

        # Complete all token states - each token now has COMPLETED terminal state
        for _, state in states:
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="completed",
                duration_ms=duration_ms,
            )

        # Register artifact (linked to first state for simplicity)
        first_state = states[0][1]
        artifact = self._recorder.register_artifact(
            run_id=self._run_id,
            state_id=first_state.state_id,
            sink_node_id=sink.node_id,
            artifact_type=sink.name,
            path=artifact_info.get("path", ""),
            content_hash=artifact_info.get("content_hash", ""),
            size_bytes=artifact_info.get("size_bytes", 0),
        )

        return artifact
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_executors.py::TestSinkExecutor -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(engine): add SinkExecutor with artifact recording"
```

---

## Task 17: RetryManager - tenacity Integration

**Files:**
- Create: `src/elspeth/engine/retry.py`
- Create: `tests/engine/test_retry.py`

### Step 1: Write the failing test

```python
# tests/engine/test_retry.py
"""Tests for RetryManager."""

import pytest


class TestRetryManager:
    """Retry logic with tenacity."""

    def test_retry_on_retryable_error(self) -> None:
        from elspeth.engine.retry import RetryManager, RetryConfig

        manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))

        call_count = 0

        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"

        result = manager.execute_with_retry(
            flaky_operation,
            is_retryable=lambda e: isinstance(e, ValueError),
        )

        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_non_retryable(self) -> None:
        from elspeth.engine.retry import RetryManager, RetryConfig

        manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))

        def failing_operation():
            raise TypeError("Not retryable")

        with pytest.raises(TypeError):
            manager.execute_with_retry(
                failing_operation,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

    def test_max_attempts_exceeded(self) -> None:
        from elspeth.engine.retry import RetryManager, RetryConfig, MaxRetriesExceeded

        manager = RetryManager(RetryConfig(max_attempts=2, base_delay=0.01))

        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(MaxRetriesExceeded) as exc_info:
            manager.execute_with_retry(
                always_fails,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

        assert exc_info.value.attempts == 2

    def test_records_attempts(self) -> None:
        from elspeth.engine.retry import RetryManager, RetryConfig

        manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))
        attempts = []

        call_count = 0

        def flaky_with_tracking():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "ok"

        result = manager.execute_with_retry(
            flaky_with_tracking,
            is_retryable=lambda e: isinstance(e, ValueError),
            on_retry=lambda attempt, error: attempts.append((attempt, str(error))),
        )

        assert len(attempts) == 1
        assert attempts[0][0] == 1

    def test_from_policy_none_returns_no_retry(self) -> None:
        """Missing policy defaults to no-retry for safety."""
        from elspeth.engine.retry import RetryConfig

        config = RetryConfig.from_policy(None)

        assert config.max_attempts == 1

    def test_from_policy_handles_malformed(self) -> None:
        """Malformed policy values are clamped to safe minimums."""
        from elspeth.engine.retry import RetryConfig

        config = RetryConfig.from_policy({
            "max_attempts": -5,  # Invalid, should clamp to 1
            "base_delay": -1,   # Invalid, should clamp to 0.01
        })

        assert config.max_attempts == 1
        assert config.base_delay >= 0.01
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_retry.py -v`
Expected: FAIL (ImportError)

### Step 3: Create retry module

```python
# src/elspeth/engine/retry.py
"""RetryManager: Retry logic with tenacity integration.

Provides configurable retry behavior for transform execution:
- Exponential backoff with jitter
- Configurable max attempts
- Retryable error filtering
- Attempt tracking for Landscape
"""

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

T = TypeVar("T")


class MaxRetriesExceeded(Exception):
    """Raised when max retry attempts are exceeded."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Max retries ({attempts}) exceeded: {last_error}")


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    max_attempts is the TOTAL number of tries, not the number of retries.
    So max_attempts=3 means: try, retry, retry (3 total).
    """

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter: float = 1.0  # seconds

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    @classmethod
    def no_retry(cls) -> "RetryConfig":
        """Factory for no-retry configuration (single attempt)."""
        return cls(max_attempts=1)

    @classmethod
    def from_policy(cls, policy: dict[str, Any] | None) -> "RetryConfig":
        """Factory from plugin policy dict with safe defaults.

        Handles missing/malformed policy gracefully.
        """
        if policy is None:
            return cls.no_retry()

        return cls(
            max_attempts=max(1, policy.get("max_attempts", 3)),
            base_delay=max(0.01, policy.get("base_delay", 1.0)),
            max_delay=max(0.1, policy.get("max_delay", 60.0)),
            jitter=max(0.0, policy.get("jitter", 1.0)),
        )


class RetryManager:
    """Manages retry logic for transform execution.

    Uses tenacity for exponential backoff with jitter.
    Integrates with Landscape for attempt tracking.

    Example:
        manager = RetryManager(RetryConfig(max_attempts=3))

        result = manager.execute_with_retry(
            operation=lambda: transform.process(row, ctx),
            is_retryable=lambda e: e.retryable,
            on_retry=lambda attempt, error: recorder.record_attempt(attempt, error),
        )
    """

    def __init__(self, config: RetryConfig) -> None:
        """Initialize with config.

        Args:
            config: Retry configuration
        """
        self._config = config

    def execute_with_retry(
        self,
        operation: Callable[[], T],
        *,
        is_retryable: Callable[[Exception], bool],
        on_retry: Callable[[int, Exception], None] | None = None,
    ) -> T:
        """Execute operation with retry logic.

        Args:
            operation: Operation to execute
            is_retryable: Function to check if error is retryable
            on_retry: Optional callback on retry (attempt, error)

        Returns:
            Result of operation

        Raises:
            MaxRetriesExceeded: If max attempts exceeded
            Exception: If non-retryable error occurs
        """
        attempt = 0
        last_error: Exception | None = None

        try:
            for attempt_state in Retrying(
                stop=stop_after_attempt(self._config.max_attempts),
                wait=wait_exponential_jitter(
                    initial=self._config.base_delay,
                    max=self._config.max_delay,
                    jitter=self._config.jitter,
                ),
                retry=retry_if_exception(is_retryable),
                reraise=False,  # Must be False so RetryError is raised, not original
            ):
                with attempt_state:
                    attempt = attempt_state.retry_state.attempt_number
                    try:
                        return operation()
                    except Exception as e:
                        last_error = e
                        if is_retryable(e) and on_retry:
                            on_retry(attempt, e)
                        raise

        except RetryError as e:
            raise MaxRetriesExceeded(attempt, last_error or e.last_attempt.exception())

        # Should not reach here
        raise RuntimeError("Unexpected state in retry loop")
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_retry.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/retry.py tests/engine/test_retry.py
git commit -m "feat(engine): add RetryManager with tenacity integration"
```

---

## Task 18: RowProcessor - Row Processing Orchestration

**Files:**
- Create: `src/elspeth/engine/processor.py`
- Create: `tests/engine/test_processor.py`

### Step 1: Write the failing test

```python
# tests/engine/test_processor.py
"""Tests for RowProcessor."""

import pytest


class TestRowProcessor:
    """Row processing through pipeline."""

    def test_process_through_transforms(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        transform1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="double",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        transform2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="add_one",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        class DoubleTransform:
            name = "double"
            node_id = transform1.node_id

            def process(self, row, ctx):
                return TransformResult.success({"value": row["value"] * 2})

        class AddOneTransform:
            name = "add_one"
            node_id = transform2.node_id

            def process(self, row, ctx):
                return TransformResult.success({"value": row["value"] + 1})

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"value": 10},
            transforms=[DoubleTransform(), AddOneTransform()],
            ctx=ctx,
        )

        # 10 * 2 = 20, 20 + 1 = 21
        assert result.final_data == {"value": 21}
        assert result.outcome == "completed"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_processor.py -v`
Expected: FAIL (ImportError)

### Step 3: Create processor module

```python
# src/elspeth/engine/processor.py
"""RowProcessor: Orchestrates row processing through pipeline.

Coordinates:
- Token creation
- Transform execution
- Gate evaluation
- Aggregation handling
- Final outcome recording
"""

from dataclasses import dataclass
from typing import Any

from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    TransformExecutor,
)
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenInfo, TokenManager
from elspeth.plugins.context import PluginContext


@dataclass
class RowResult:
    """Result of processing a row through the pipeline."""

    token: TokenInfo  # Preserve full token identity, not just IDs
    final_data: dict[str, Any]
    outcome: str  # completed, routed, forked, consumed, failed
    sink_name: str | None = None  # Set when outcome is "routed"

    @property
    def token_id(self) -> str:
        return self.token.token_id

    @property
    def row_id(self) -> str:
        return self.token.row_id


class RowProcessor:
    """Processes rows through the transform pipeline.

    Handles:
    1. Creating initial tokens from source rows
    2. Executing transforms in sequence
    3. Evaluating gates for routing decisions
    4. Accepting rows into aggregations
    5. Recording final outcomes

    Example:
        processor = RowProcessor(recorder, span_factory, run_id, source_node_id)

        result = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[transform1, transform2],
            ctx=ctx,
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
        source_node_id: str,
        *,
        edge_map: dict[tuple[str, str], str] | None = None,
    ) -> None:
        """Initialize processor.

        Args:
            recorder: Landscape recorder
            span_factory: Span factory for tracing
            run_id: Current run ID
            source_node_id: Source node ID
            edge_map: Map of (node_id, label) -> edge_id
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._source_node_id = source_node_id

        self._token_manager = TokenManager(recorder)
        self._transform_executor = TransformExecutor(recorder, span_factory)
        self._gate_executor = GateExecutor(recorder, span_factory, edge_map)
        self._aggregation_executor = AggregationExecutor(
            recorder, span_factory, run_id
        )

    def process_row(
        self,
        row_index: int,
        row_data: dict[str, Any],
        transforms: list[Any],
        ctx: PluginContext,
    ) -> RowResult:
        """Process a row through all transforms.

        NOTE: This implementation handles LINEAR pipelines only. For DAG support
        (fork/join), this needs a work queue that processes child tokens from forks.
        Currently fork_to_paths returns "forked" and the caller must handle the children.

        Args:
            row_index: Position in source
            row_data: Initial row data
            transforms: List of transform plugins
            ctx: Plugin context

        Returns:
            RowResult with final outcome
        """
        # Create initial token
        token = self._token_manager.create_initial_token(
            run_id=self._run_id,
            source_node_id=self._source_node_id,
            row_index=row_index,
            row_data=row_data,
        )

        with self._spans.row_span(token.row_id, token.token_id):
            current_token = token
            step = 0  # Track position in pipeline - RowProcessor is the authority

            for transform in transforms:
                step += 1  # Advance step before each transform

                # Check transform type and execute accordingly
                if hasattr(transform, "evaluate"):
                    # Gate transform
                    outcome = self._gate_executor.execute_gate(
                        gate=transform,
                        token=current_token,
                        ctx=ctx,
                        step_in_pipeline=step,
                        token_manager=self._token_manager,
                    )
                    current_token = outcome.updated_token

                    if outcome.result.action.kind == "route_to_sink":
                        return RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome="routed",
                            sink_name=outcome.sink_name,  # GateOutcome provides this!
                        )
                    elif outcome.result.action.kind == "fork_to_paths":
                        # NOTE: For full DAG support, we'd push child_tokens to a work queue
                        # and continue processing them. For now, return "forked".
                        return RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome="forked",
                        )

                elif hasattr(transform, "accept"):
                    # Aggregation transform
                    accept_result = self._aggregation_executor.accept(
                        aggregation=transform,
                        token=current_token,
                        ctx=ctx,
                        step_in_pipeline=step,
                    )

                    if accept_result.trigger:
                        self._aggregation_executor.flush(
                            aggregation=transform,
                            ctx=ctx,
                            trigger_reason="threshold",
                            step_in_pipeline=step,
                        )

                    return RowResult(
                        token=current_token,
                        final_data=current_token.row_data,
                        outcome="consumed",
                    )

                else:
                    # Regular transform
                    result, current_token = self._transform_executor.execute_transform(
                        transform=transform,
                        token=current_token,
                        ctx=ctx,
                        step_in_pipeline=step,
                    )

                    if result.status == "error":
                        return RowResult(
                            token=current_token,
                            final_data=current_token.row_data,
                            outcome="failed",
                        )

            return RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome="completed",
            )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_processor.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor.py
git commit -m "feat(engine): add RowProcessor for pipeline orchestration"
```

---

## Task 19: Orchestrator - Full Run Lifecycle

**Files:**
- Create: `src/elspeth/engine/orchestrator.py`
- Create: `tests/engine/test_orchestrator.py`

### Step 1: Write the failing test

```python
# tests/engine/test_orchestrator.py
"""Tests for Orchestrator."""

import pytest


class TestOrchestrator:
    """Full run orchestration."""

    def test_run_simple_pipeline(self) -> None:
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int
            doubled: int

        class ListSource:
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class DoubleTransform:
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row, ctx):
                return TransformResult.success({
                    "value": row["value"],
                    "doubled": row["value"] * 2,
                })

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []  # Instance attribute, not class attribute

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = DoubleTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config)

        assert run_result.status == "completed"
        assert run_result.rows_processed == 3
        assert len(sink.results) == 3
        assert sink.results[0] == {"value": 1, "doubled": 2}

    def test_run_with_gate_routing(self) -> None:
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import GateResult, RoutingAction, TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class ThresholdGate:
            name = "threshold"
            input_schema = RowSchema
            output_schema = RowSchema

            def evaluate(self, row, ctx):
                if row["value"] > 50:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink("high"),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        class CollectSink:
            name = "collect"

            def __init__(self):
                self.results = []  # Instance attribute, not class attribute

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

        source = ListSource([{"value": 10}, {"value": 100}, {"value": 30}])
        gate = ThresholdGate()
        default_sink = CollectSink()
        high_sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[gate],
            sinks={"default": default_sink, "high": high_sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config)

        assert run_result.status == "completed"
        # value=10 and value=30 go to default, value=100 goes to high
        assert len(default_sink.results) == 2
        assert len(high_sink.results) == 1
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_orchestrator.py -v`
Expected: FAIL (ImportError)

### Step 3: Create orchestrator module

```python
# src/elspeth/engine/orchestrator.py
"""Orchestrator: Full run lifecycle management.

Coordinates:
- Run initialization
- Source loading
- Row processing
- Sink writing
- Run completion
"""

from dataclasses import dataclass, field
from typing import Any

from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.processor import RowProcessor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext
from elspeth.plugins.enums import NodeType


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""

    source: Any  # SourceProtocol
    transforms: list[Any]  # List of transform/gate plugins
    sinks: dict[str, Any]  # sink_name -> SinkProtocol
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: str  # completed, failed
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed: int


class Orchestrator:
    """Orchestrates full pipeline runs.

    Manages the complete lifecycle:
    1. Begin run in Landscape
    2. Register all nodes (and set node_id on each plugin instance)
    3. Load rows from source
    4. Process rows through transforms
    5. Write to sinks
    6. Complete run

    NOTE on node_id: Plugin protocols (TransformProtocol, etc.) don't
    define node_id as an attribute. The Orchestrator sets node_id on
    each plugin instance AFTER registering it with Landscape:

        node = recorder.register_node(...)
        transform.node_id = node.node_id  # Set by Orchestrator

    This allows executors to access node_id without requiring plugins
    to know their node_id at construction time.

    Example:
        orchestrator = Orchestrator(db)

        config = PipelineConfig(
            source=csv_source,
            transforms=[transform1, gate1, transform2],
            sinks={"default": csv_sink, "flagged": review_sink},
        )

        result = orchestrator.run(config)
    """

    def __init__(
        self,
        db: LandscapeDB,
        *,
        canonical_version: str = "sha256-rfc8785-v1",
    ) -> None:
        """Initialize orchestrator.

        Args:
            db: Landscape database
            canonical_version: Canonical hash version
        """
        self._db = db
        self._canonical_version = canonical_version
        self._span_factory = SpanFactory()

    def run(self, config: PipelineConfig) -> RunResult:
        """Execute a pipeline run.

        Args:
            config: Pipeline configuration

        Returns:
            RunResult with execution summary
        """
        recorder = LandscapeRecorder(self._db)

        # Begin run
        run = recorder.begin_run(
            config=config.config,
            canonical_version=self._canonical_version,
        )

        try:
            with self._span_factory.run_span(run.run_id):
                result = self._execute_run(recorder, run.run_id, config)

            # Complete run
            recorder.complete_run(run.run_id, status="completed")
            result.status = "completed"
            return result

        except Exception as e:
            recorder.complete_run(run.run_id, status="failed")
            raise

    def _execute_run(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        config: PipelineConfig,
    ) -> RunResult:
        """Execute the run (internal)."""

        # Register source node
        source_node = recorder.register_node(
            run_id=run_id,
            plugin_name=config.source.name,
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            sequence=0,
        )

        # Register transform nodes and track gates for sink edge registration
        edge_map: dict[tuple[str, str], str] = {}
        prev_node_id = source_node.node_id
        gate_node_ids: list[str] = []  # Track gates that may route to sinks

        for i, transform in enumerate(config.transforms):
            is_gate = hasattr(transform, "evaluate")
            node_type = NodeType.GATE if is_gate else NodeType.TRANSFORM
            node = recorder.register_node(
                run_id=run_id,
                plugin_name=transform.name,
                node_type=node_type,
                plugin_version="1.0.0",
                config={},
                sequence=i + 1,
            )
            # Set node_id on plugin (see class docstring for why this is needed)
            transform.node_id = node.node_id

            # Track gates - they may route to any sink
            if is_gate:
                gate_node_ids.append(node.node_id)

            # Register continue edge
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=prev_node_id,
                to_node_id=node.node_id,
                label="continue",
                mode="move",
            )
            edge_map[(prev_node_id, "continue")] = edge.edge_id
            prev_node_id = node.node_id

        # Register sink nodes
        sink_nodes: dict[str, Any] = {}
        for sink_name, sink in config.sinks.items():
            node = recorder.register_node(
                run_id=run_id,
                plugin_name=sink.name,
                node_type=NodeType.SINK,
                plugin_version="1.0.0",
                config={},
            )
            sink.node_id = node.node_id
            sink_nodes[sink_name] = node

            # Register edge from last transform to sink (for continue path)
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=prev_node_id,
                to_node_id=node.node_id,
                label=sink_name,
                mode="move",
            )
            edge_map[(prev_node_id, sink_name)] = edge.edge_id

            # CRITICAL: Register edges from ALL gates to this sink
            # Gates at any position may route_to_sink, not just the last node
            for gate_node_id in gate_node_ids:
                if gate_node_id != prev_node_id:  # Don't duplicate
                    gate_edge = recorder.register_edge(
                        run_id=run_id,
                        from_node_id=gate_node_id,
                        to_node_id=node.node_id,
                        label=sink_name,
                        mode="move",
                    )
                    edge_map[(gate_node_id, sink_name)] = gate_edge.edge_id

        # Create context
        ctx = PluginContext(
            run_id=run_id,
            config=config.config,
            landscape=recorder,
        )

        # Create processor
        processor = RowProcessor(
            recorder=recorder,
            span_factory=self._span_factory,
            run_id=run_id,
            source_node_id=source_node.node_id,
            edge_map=edge_map,
        )

        # Process rows - CRITICAL: Buffer TOKENS, not dicts, to preserve identity
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.tokens import TokenInfo

        rows_processed = 0
        rows_succeeded = 0
        rows_failed = 0
        rows_routed = 0
        pending_tokens: dict[str, list[TokenInfo]] = {name: [] for name in config.sinks}

        try:
            with self._span_factory.source_span(config.source.name):
                for row_index, row_data in enumerate(config.source.load(ctx)):
                    rows_processed += 1

                    result = processor.process_row(
                        row_index=row_index,
                        row_data=row_data,
                        transforms=config.transforms,
                        ctx=ctx,
                    )

                    if result.outcome == "completed":
                        rows_succeeded += 1
                        # Preserve full TokenInfo, not just dict
                        pending_tokens["default"].append(result.token)
                    elif result.outcome == "routed":
                        rows_routed += 1
                        # Use the actual sink_name from RowResult (not guessing!)
                        if result.sink_name and result.sink_name in config.sinks:
                            pending_tokens[result.sink_name].append(result.token)
                        else:
                            # Routed to unknown sink - this is a configuration error
                            rows_failed += 1
                    elif result.outcome == "failed":
                        rows_failed += 1

            # Write to sinks using SinkExecutor with proper audit
            sink_executor = SinkExecutor(recorder, self._span_factory, run_id)
            step = len(config.transforms) + 1  # Sinks are after all transforms

            for sink_name, tokens in pending_tokens.items():
                if tokens and sink_name in config.sinks:
                    sink = config.sinks[sink_name]
                    # CRITICAL: Pass TokenInfo list, not dicts, for proper node_states
                    sink_executor.write(
                        sink=sink,
                        tokens=tokens,
                        ctx=ctx,
                        step_in_pipeline=step,
                    )

        finally:
            # Close source and all sinks - ALWAYS, even on exception
            config.source.close()
            for sink in config.sinks.values():
                sink.close()

        return RunResult(
            run_id=run_id,
            status="running",  # Will be updated
            rows_processed=rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            rows_routed=rows_routed,
        )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_orchestrator.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "feat(engine): add Orchestrator for full run lifecycle"
```

---

## Task 20: Engine Module Exports and Final Verification

**Files:**
- Modify: `src/elspeth/engine/__init__.py`
- Create: `tests/engine/test_integration.py`

### Step 1: Write the integration test

```python
# tests/engine/test_integration.py
"""Integration tests for SDA Engine."""

import pytest


class TestEngineIntegration:
    """Full engine integration tests."""

    def test_can_import_all_components(self) -> None:
        from elspeth.engine import (
            Orchestrator,
            PipelineConfig,
            RowProcessor,
            RowResult,
            RunResult,
            SpanFactory,
            TokenInfo,
            TokenManager,
        )

        assert Orchestrator is not None
        assert RowProcessor is not None
        assert SpanFactory is not None

    def test_full_pipeline_with_audit(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data):
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class IncrementTransform:
            name = "increment"
            input_schema = RowSchema
            output_schema = RowSchema

            def process(self, row, ctx):
                return TransformResult.success({"value": row["value"] + 1})

        class MemorySink:
            name = "memory"
            results = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

        source = ListSource([{"value": i} for i in range(5)])
        transform = IncrementTransform()
        sink = MemorySink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config)

        # Check result
        assert result.status == "completed"
        assert result.rows_processed == 5
        assert result.rows_succeeded == 5

        # Check sink received transformed data
        assert len(sink.results) == 5
        assert sink.results[0] == {"value": 1}  # 0 + 1
        assert sink.results[4] == {"value": 5}  # 4 + 1

        # Check audit trail
        recorder = LandscapeRecorder(db)
        run = recorder.get_run(result.run_id)
        assert run is not None
        assert run.status == "completed"

        nodes = recorder.get_nodes(result.run_id)
        assert len(nodes) >= 3  # source, transform, sink

        # CRITICAL AUDIT VERIFICATION: Every token must reach a terminal state
        # This is the Attributability Test - no silent drops allowed!
        rows = recorder.get_rows(result.run_id)
        assert len(rows) == 5, "Should have 5 rows in audit trail"

        # Each row should have a token
        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            assert len(tokens) >= 1, f"Row {row.row_id} should have at least one token"

            # Each token should have reached the sink (completed node_state at sink)
            for token in tokens:
                states = recorder.get_node_states(token.token_id)
                assert len(states) >= 2, (
                    f"Token {token.token_id} should have states at transform AND sink"
                )

                # Find sink node and verify token reached it
                sink_states = [s for s in states if s.node_id == sink.node_id]
                assert len(sink_states) == 1, (
                    f"Token {token.token_id} should have exactly one state at sink"
                )
                assert sink_states[0].status == "completed", (
                    f"Token {token.token_id} state at sink should be 'completed'"
                )

        # Verify artifact was recorded
        artifacts = recorder.get_artifacts(result.run_id)
        assert len(artifacts) >= 1, "Should have at least one artifact from sink"

    def test_audit_spine_intact(self) -> None:
        """THE audit spine test: proves the chassis doesn't wobble.

        A simple run must produce:
        - runs: 1 record, status=completed
        - nodes: source + transform + sink (3 minimum)
        - tokens: at least the initial tokens
        - node_states: at least one per node type exercised
        - artifacts: 1 record for the sink

        If this test fails, something fundamental is broken.
        """
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class SimpleSource:
            name = "simple_source"
            def load(self, ctx):
                yield {"x": 1}
                yield {"x": 2}
            def close(self):
                pass

        class PassThrough:
            name = "passthrough"
            def process(self, row, ctx):
                return TransformResult.success(row)

        class RecordingSink:
            name = "recorder"
            def write(self, rows, ctx):
                return {"path": "memory://", "size_bytes": 0, "content_hash": "none"}

        config = PipelineConfig(
            source=SimpleSource(),
            transforms=[PassThrough()],
            sinks={"default": RecordingSink()},
        )

        result = Orchestrator(db).run(config)
        recorder = LandscapeRecorder(db)

        # === AUDIT SPINE ASSERTIONS ===

        # 1. runs: exactly 1, status=completed
        run = recorder.get_run(result.run_id)
        assert run is not None, "Run must exist"
        assert run.status == "completed", "Run must be completed"

        # 2. nodes: source + transform + sink = 3
        nodes = recorder.get_nodes(result.run_id)
        node_types = {n.node_type for n in nodes}
        assert "source" in node_types, "Must have source node"
        assert "transform" in node_types, "Must have transform node"
        assert "sink" in node_types, "Must have sink node"
        assert len(nodes) == 3, f"Expected 3 nodes, got {len(nodes)}"

        # 3. tokens: at least 2 (one per source row)
        rows = recorder.get_rows(result.run_id)
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
        token_count = sum(len(recorder.get_tokens(r.row_id)) for r in rows)
        assert token_count >= 2, f"Expected at least 2 tokens, got {token_count}"

        # 4. node_states: each token must have states at transform AND sink
        for row in rows:
            for token in recorder.get_tokens(row.row_id):
                states = recorder.get_node_states(token.token_id)
                state_node_types = {
                    next(n.node_type for n in nodes if n.node_id == s.node_id)
                    for s in states
                }
                assert "transform" in state_node_types, (
                    f"Token {token.token_id} missing transform state"
                )
                assert "sink" in state_node_types, (
                    f"Token {token.token_id} missing sink state"
                )
                # All states must be completed (not open, not failed)
                for state in states:
                    assert state.status == "completed", (
                        f"State {state.state_id} should be completed, got {state.status}"
                    )

        # 5. artifacts: exactly 1 from sink
        artifacts = recorder.get_artifacts(result.run_id)
        assert len(artifacts) == 1, f"Expected 1 artifact, got {len(artifacts)}"

    def test_audit_spine_with_routing(self) -> None:
        """Audit spine with routing: proves routed tokens are tracked.

        When a gate routes to a non-default sink:
        - routing_events must exist
        - node_states at the routed sink must be completed
        """
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()

        class MixedSource:
            name = "mixed"
            def load(self, ctx):
                yield {"val": 10}   # below threshold -> default
                yield {"val": 100}  # above threshold -> high
                yield {"val": 20}   # below threshold -> default
            def close(self):
                pass

        class ThresholdGate:
            name = "threshold"
            def evaluate(self, row, ctx):
                if row["val"] >= 50:
                    return GateResult(row=row, action=RoutingAction.route_to_sink("high"))
                return GateResult(row=row, action=RoutingAction.continue_())

        class Collector:
            name = "collector"

            def __init__(self):
                self.items = []  # Instance attribute, not class attribute

            def write(self, rows, ctx):
                self.items.extend(rows)
                return {"path": "mem://", "size_bytes": 0, "content_hash": "x"}

        default_sink = Collector()
        high_sink = Collector()

        config = PipelineConfig(
            source=MixedSource(),
            transforms=[ThresholdGate()],
            sinks={"default": default_sink, "high": high_sink},
        )

        result = Orchestrator(db).run(config)
        recorder = LandscapeRecorder(db)

        # Verify routing happened correctly
        assert len(default_sink.items) == 2, "2 rows should go to default"
        assert len(high_sink.items) == 1, "1 row should go to high"

        # === ROUTING AUDIT ASSERTIONS ===

        # Get the gate node
        nodes = recorder.get_nodes(result.run_id)
        gate_node = next(n for n in nodes if n.node_type == "gate")

        # Find the routed token (val=100)
        rows = recorder.get_rows(result.run_id)
        routed_row = next(r for r in rows if recorder.get_row_data(r.row_id)["val"] == 100)
        routed_token = recorder.get_tokens(routed_row.row_id)[0]

        # The routed token must have a routing_event at the gate
        gate_state = next(
            s for s in recorder.get_node_states(routed_token.token_id)
            if s.node_id == gate_node.node_id
        )
        routing_events = recorder.get_routing_events(gate_state.state_id)
        assert len(routing_events) >= 1, "Routed token must have routing_event"

        # The routed token must reach the high sink
        high_sink_node = next(n for n in nodes if n.plugin_name == "collector" and n != next(n2 for n2 in nodes if n2.plugin_name == "collector"))
        # Simpler: just verify token has state at some sink node
        token_states = recorder.get_node_states(routed_token.token_id)
        sink_states = [s for s in token_states if any(n.node_id == s.node_id and n.node_type == "sink" for n in nodes)]
        assert len(sink_states) == 1, "Routed token must reach exactly one sink"
        assert sink_states[0].status == "completed"


class TestNoSilentAuditLoss:
    """Lock down the 'no silent audit loss' principle.

    These tests exist to prevent a whole class of future incidents
    where evidence is silently discarded.
    """

    def test_missing_edge_raises_not_skips(self) -> None:
        """GateExecutor MUST raise when edge is missing, not skip recording.

        This is the critical test that prevents silent audit loss.
        If someone changes GateExecutor to silently skip missing edges,
        this test will catch it.
        """
        import pytest
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor, MissingEdgeError
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="router",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )

        class RouterGate:
            name = "router"
            node_id = gate_node.node_id

            def evaluate(self, row, ctx):
                return GateResult(
                    row=row,
                    action=RoutingAction.route_to_sink("unregistered_sink"),
                )

        gate = RouterGate()
        ctx = PluginContext(run_id=run.run_id, config={})

        # CRITICAL: empty edge_map means NO edges registered
        executor = GateExecutor(recorder, SpanFactory(), edge_map={})

        token = TokenInfo(row_id="r1", token_id="t1", row_data={"x": 1})
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        # THE CRITICAL ASSERTION: Must raise, not silently skip
        with pytest.raises(MissingEdgeError) as exc:
            executor.execute_gate(gate=gate, token=token, ctx=ctx, step_in_pipeline=1)

        # Verify the error is informative
        assert "unregistered_sink" in str(exc.value)
        assert gate_node.node_id in str(exc.value)

    def test_missing_edge_error_is_not_catchable_silently(self) -> None:
        """MissingEdgeError should NOT be a subclass of common ignored exceptions.

        Ensures that no one can accidentally catch and suppress this error
        with a broad except clause like `except ValueError`.
        """
        from elspeth.engine.executors import MissingEdgeError

        # MissingEdgeError inherits from Exception, not ValueError/KeyError/etc
        assert issubclass(MissingEdgeError, Exception)
        assert not issubclass(MissingEdgeError, (ValueError, KeyError, TypeError))

        # The error must be explicitly named to catch
        error = MissingEdgeError(node_id="n1", label="sink_x")
        assert "Audit trail would be incomplete" in str(error)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_integration.py -v`
Expected: FAIL (ImportError)

### Step 3: Update engine module exports

```python
# src/elspeth/engine/__init__.py
"""SDA Engine: Orchestration with complete audit trails.

This module provides the execution engine for ELSPETH pipelines:
- Orchestrator: Full run lifecycle management
- RowProcessor: Row-by-row processing through transforms
- TokenManager: Token identity through forks/joins
- SpanFactory: OpenTelemetry integration
- RetryManager: Retry logic with tenacity

Example:
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig

    db = LandscapeDB.from_url("sqlite:///audit.db")

    config = PipelineConfig(
        source=csv_source,
        transforms=[transform1, gate1],
        sinks={"default": output_sink},
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config)
"""

from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    MissingEdgeError,
    SinkExecutor,
    TransformExecutor,
)
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig, RunResult
from elspeth.engine.processor import RowProcessor, RowResult
from elspeth.engine.retry import MaxRetriesExceeded, RetryConfig, RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenInfo, TokenManager

__all__ = [
    # Orchestration
    "Orchestrator",
    "PipelineConfig",
    "RunResult",
    # Processing
    "RowProcessor",
    "RowResult",
    # Tokens
    "TokenManager",
    "TokenInfo",
    # Executors
    "TransformExecutor",
    "GateExecutor",
    "AggregationExecutor",
    "SinkExecutor",
    "MissingEdgeError",
    # Retry
    "RetryManager",
    "RetryConfig",
    "MaxRetriesExceeded",
    # Tracing
    "SpanFactory",
]
```

### Step 4: Run all tests

Run: `pytest tests/engine/ tests/core/landscape/ -v`
Expected: ALL PASS

### Step 5: Final commit

```bash
git add src/elspeth/engine/__init__.py tests/engine/test_integration.py
git commit -m "feat(engine): export public API and add integration tests"
```

---

# Phase 3 Complete

**All Tasks:**
1. ✅ LandscapeSchema - SQLAlchemy table definitions
2. ✅ LandscapeDB - Database connection manager
3. ✅ LandscapeRecorder - Run management
4. ✅ LandscapeRecorder - Node and edge registration
5. ✅ LandscapeRecorder - Row and token creation
6. ✅ LandscapeRecorder - NodeState recording
7. ✅ LandscapeRecorder - Routing events
8. ✅ LandscapeRecorder - Batch management
9. ✅ LandscapeRecorder - Artifact registration
10. ✅ Landscape module exports
11. ✅ SpanFactory - OpenTelemetry integration
12. ✅ TokenManager - High-level token operations
13. ✅ TransformExecutor - Audit-wrapped transform execution
14. ✅ GateExecutor - Routing with audit
15. ✅ AggregationExecutor - Batch tracking
16. ✅ SinkExecutor - Artifact recording
17. ✅ RetryManager - tenacity integration
18. ✅ RowProcessor - Row processing orchestration
19. ✅ Orchestrator - Full run lifecycle
20. ✅ Engine module exports and integration tests

---

## Final Verification

```bash
# Run all Phase 3 tests
pytest tests/core/landscape/ tests/engine/ -v

# Run full test suite
pytest -v

# Type check
mypy src/elspeth/core/landscape src/elspeth/engine

# Lint
ruff check src/elspeth/core/landscape src/elspeth/engine
```

---

**Final commit:**

```bash
git add docs/plans/2026-01-12-phase3-sda-engine.md
git commit -m "docs: complete Phase 3 SDA Engine implementation plan"
```
