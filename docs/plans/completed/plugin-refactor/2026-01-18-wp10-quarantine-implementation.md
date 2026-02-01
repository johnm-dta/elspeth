# WP-10: Quarantine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `RowOutcome.QUARANTINED` reachable as a terminal state for rows that fail validation or transform processing, with complete audit trail.

**Architecture:** Quarantine is the intentional, audited rejection of a row due to data quality issues (not bugs). Sources already quarantine at ingestion; this WP extends quarantine to transform errors when `on_error="discard"`. The executor returns error routing info to the processor, which returns the correct terminal outcome (QUARANTINED for discards, ROUTED for error sinks).

**Tech Stack:** Existing `LandscapeRecorder.record_validation_error()`, `PluginContext`, `TransformExecutor`, `RowProcessor`.

---

## Background: Current State Analysis

### What Already Exists

| Component | Location | Status |
|-----------|----------|--------|
| `RowOutcome.QUARANTINED` | enums.py:133 | ✅ Enum exists |
| `LandscapeRecorder.record_validation_error()` | recorder.py:1898-1938 | ✅ Records to audit trail |
| `PluginContext.record_validation_error()` | context.py:127-140 | ✅ Plugin API exists |
| `ValidationErrorToken` | context.py:31-38 | ✅ Tracking dataclass |
| Source-level quarantine | csv_source.py, json_source.py | ✅ Sources quarantine invalid rows |
| `validation_errors_table` | landscape schema | ✅ Audit table exists |

### What's Missing

| Gap | Current Behavior | Required Behavior |
|-----|------------------|-------------------|
| Transform error outcome | Always returns `FAILED` | Return `QUARANTINED` if discarded, `ROUTED` if sent to error sink |
| Processor doesn't know error fate | Executor handles routing as side effect | Executor returns routing info to processor |
| No distinction FAILED vs QUARANTINED | Both are "error happened" | FAILED = unexpected, QUARANTINED = intentional rejection |

### The Problem in Detail

Current flow when transform returns error:

```
1. transform.process() → TransformResult(status="error")
2. TransformExecutor:
   - Records error in node_state
   - If on_error is None: raise RuntimeError ← Correct
   - If on_error == "discard": ctx.record_transform_error() ← Side effect
   - If on_error == sink_name: ctx.route_to_sink() ← Side effect
3. Processor sees status="error" → returns FAILED ← WRONG!
```

The processor doesn't know if the error was:
- Handled and discarded → should be QUARANTINED
- Routed to error sink → should be ROUTED

### Design Decision

Add `error_sink` field to the executor return to communicate error handling outcome:

```python
# New return signature
def execute_transform(...) -> tuple[TransformResult, TokenInfo, str | None]:
    """
    Returns:
        result: The TransformResult
        token: Updated token
        error_sink: None if success, "discard" if quarantined, or sink_name if routed
    """
```

Then processor:
- `error_sink is None` + `status == "success"` → continue processing
- `error_sink == "discard"` → return `QUARANTINED`
- `error_sink == sink_name` → return `ROUTED` with sink_name

---

## Task 1: Update TransformExecutor Return Signature

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Test: `tests/engine/test_executors.py`

### Step 1: Write failing test for new return signature

```python
# tests/engine/test_executors.py - add to TestTransformExecutor class

def test_execute_transform_returns_error_sink_on_discard(
    self, db: LandscapeDB, run: Run
) -> None:
    """When transform errors with on_error='discard', returns error_sink='discard'."""
    from elspeth.contracts import Determinism, NodeType
    from elspeth.plugins.base import BaseTransform
    from elspeth.plugins.results import TransformResult
    from elspeth.plugins.schemas import DynamicSchema

    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()

    node = recorder.register_node(
        run_id=run.run_id,
        node_id="transform_1",
        plugin_name="error_transform",
        node_type=NodeType.TRANSFORM,
    )

    class ErrorTransform(BaseTransform):
        name = "error_transform"
        input_schema = DynamicSchema
        output_schema = DynamicSchema
        determinism = Determinism.DETERMINISTIC
        plugin_version = "1.0.0"
        _on_error = "discard"  # Configured to discard errors

        def __init__(self):
            self.node_id = node.node_id

        def process(self, row, ctx):
            return TransformResult.error({"reason": "intentional_failure"})

    transform = ErrorTransform()
    executor = TransformExecutor(recorder, span_factory)

    token = TokenInfo(
        row_id="row_1",
        token_id="token_1",
        row_data={"value": 42},
    )
    ctx = PluginContext(run_id=run.run_id, config={})

    result, updated_token, error_sink = executor.execute_transform(
        transform=transform,
        token=token,
        ctx=ctx,
        step_in_pipeline=1,
    )

    assert result.status == "error"
    assert error_sink == "discard"


def test_execute_transform_returns_error_sink_name(
    self, db: LandscapeDB, run: Run
) -> None:
    """When transform errors with on_error=sink_name, returns that sink name."""
    from elspeth.contracts import Determinism, NodeType
    from elspeth.plugins.base import BaseTransform
    from elspeth.plugins.results import TransformResult
    from elspeth.plugins.schemas import DynamicSchema

    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()

    node = recorder.register_node(
        run_id=run.run_id,
        node_id="transform_1",
        plugin_name="error_transform",
        node_type=NodeType.TRANSFORM,
    )

    class ErrorTransform(BaseTransform):
        name = "error_transform"
        input_schema = DynamicSchema
        output_schema = DynamicSchema
        determinism = Determinism.DETERMINISTIC
        plugin_version = "1.0.0"
        _on_error = "error_sink"  # Configured to route to error_sink

        def __init__(self):
            self.node_id = node.node_id

        def process(self, row, ctx):
            return TransformResult.error({"reason": "intentional_failure"})

    transform = ErrorTransform()
    executor = TransformExecutor(recorder, span_factory)

    token = TokenInfo(
        row_id="row_1",
        token_id="token_1",
        row_data={"value": 42},
    )
    ctx = PluginContext(run_id=run.run_id, config={})

    result, updated_token, error_sink = executor.execute_transform(
        transform=transform,
        token=token,
        ctx=ctx,
        step_in_pipeline=1,
    )

    assert result.status == "error"
    assert error_sink == "error_sink"


def test_execute_transform_returns_none_error_sink_on_success(
    self, db: LandscapeDB, run: Run
) -> None:
    """On success, error_sink is None."""
    from elspeth.contracts import Determinism, NodeType
    from elspeth.plugins.base import BaseTransform
    from elspeth.plugins.results import TransformResult
    from elspeth.plugins.schemas import DynamicSchema

    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()

    node = recorder.register_node(
        run_id=run.run_id,
        node_id="transform_1",
        plugin_name="success_transform",
        node_type=NodeType.TRANSFORM,
    )

    class SuccessTransform(BaseTransform):
        name = "success_transform"
        input_schema = DynamicSchema
        output_schema = DynamicSchema
        determinism = Determinism.DETERMINISTIC
        plugin_version = "1.0.0"

        def __init__(self):
            self.node_id = node.node_id

        def process(self, row, ctx):
            return TransformResult.success({**row, "processed": True})

    transform = SuccessTransform()
    executor = TransformExecutor(recorder, span_factory)

    token = TokenInfo(
        row_id="row_1",
        token_id="token_1",
        row_data={"value": 42},
    )
    ctx = PluginContext(run_id=run.run_id, config={})

    result, updated_token, error_sink = executor.execute_transform(
        transform=transform,
        token=token,
        ctx=ctx,
        step_in_pipeline=1,
    )

    assert result.status == "success"
    assert error_sink is None
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/engine/test_executors.py::TestTransformExecutor::test_execute_transform_returns_error_sink_on_discard -v
pytest tests/engine/test_executors.py::TestTransformExecutor::test_execute_transform_returns_error_sink_name -v
pytest tests/engine/test_executors.py::TestTransformExecutor::test_execute_transform_returns_none_error_sink_on_success -v
```

Expected: FAIL with "cannot unpack" (wrong number of return values)

### Step 3: Update TransformExecutor.execute_transform() signature

```python
# src/elspeth/engine/executors.py - update execute_transform method

    def execute_transform(
        self,
        transform: TransformProtocol,
        token: TokenInfo,
        ctx: PluginContext,
        step_in_pipeline: int,
    ) -> tuple[TransformResult, TokenInfo, str | None]:
        """Execute a transform with full audit recording and error routing.

        This method handles a SINGLE ATTEMPT. Retry logic is the caller's
        responsibility (e.g., RetryManager wraps this for retryable transforms).
        Each attempt gets its own node_state record with attempt number tracked
        by the caller.

        Error Routing:
        - TransformResult.error() is a LEGITIMATE processing failure
        - Routes to configured sink via transform._on_error
        - RuntimeError if transform errors without on_error config
        - Exceptions are BUGS and propagate (not routed)

        Args:
            transform: Transform plugin to execute
            token: Current token with row data
            ctx: Plugin context
            step_in_pipeline: Current position in DAG (Orchestrator is authority)

        Returns:
            Tuple of:
            - TransformResult with audit fields
            - Updated TokenInfo
            - Error sink: None if success, "discard" if quarantined, or sink_name if routed

        Raises:
            Exception: Re-raised from transform.process() after recording failure
            RuntimeError: Transform returned error but has no on_error configured
        """
        assert transform.node_id is not None, "node_id must be set by orchestrator"
        input_hash = stable_hash(token.row_data)
        error_sink: str | None = None

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
                error: ExecutionError = {
                    "exception": str(e),
                    "type": type(e).__name__,
                }
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status="failed",
                    duration_ms=duration_ms,
                    error=error,
                )
                raise

        # Populate audit fields
        result.input_hash = input_hash
        result.output_hash = stable_hash(result.row) if result.row else None
        result.duration_ms = duration_ms

        # Complete node state
        if result.status == "success":
            # TransformResult.success() always sets row - this is a contract
            assert result.row is not None, "success status requires row data"
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
            # This is a LEGITIMATE processing failure, not a bug
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="failed",
                duration_ms=duration_ms,
                error=result.reason,
            )

            # Handle error routing - _on_error is part of TransformProtocol
            on_error = transform._on_error

            if on_error is None:
                raise RuntimeError(
                    f"Transform '{transform.name}' returned error but has no on_error "
                    f"configured. Either configure on_error or fix the transform to not "
                    f"return errors for this input. Error: {result.reason}"
                )

            # Set error_sink for processor to use
            error_sink = on_error

            # Record error event (always, even for discard - audit completeness)
            ctx.record_transform_error(
                token_id=token.token_id,
                transform_id=transform.name,
                row=token.row_data,
                error_details=result.reason or {},
                destination=on_error,
            )

            # Route to sink if not discarding
            if on_error != "discard":
                ctx.route_to_sink(
                    sink_name=on_error,
                    row=token.row_data,
                    metadata={"transform_error": result.reason},
                )

            updated_token = token

        return result, updated_token, error_sink
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/engine/test_executors.py::TestTransformExecutor -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/executors.py tests/engine/test_executors.py
git commit -m "feat(executor): return error_sink from execute_transform (WP-10 Task 1)

- Add third return value: error_sink (None, 'discard', or sink_name)
- Processor can now determine correct outcome (QUARANTINED vs ROUTED)
- Existing error handling logic preserved

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Update Processor to Handle Error Routing

**Files:**
- Modify: `src/elspeth/engine/processor.py`
- Test: `tests/engine/test_processor.py`

### Step 1: Write failing test for QUARANTINED outcome

```python
# tests/engine/test_processor.py - add to TestRowProcessorTransforms class

def test_transform_error_with_discard_returns_quarantined(
    self, db: LandscapeDB, run: Run
) -> None:
    """Transform error with on_error='discard' should return QUARANTINED."""
    from elspeth.contracts import Determinism, NodeType, RowOutcome
    from elspeth.plugins.base import BaseTransform
    from elspeth.plugins.results import TransformResult
    from elspeth.plugins.schemas import DynamicSchema

    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()

    source_node = recorder.register_node(
        run_id=run.run_id,
        node_id="source_1",
        plugin_name="test_source",
        node_type=NodeType.SOURCE,
    )
    transform_node = recorder.register_node(
        run_id=run.run_id,
        node_id="transform_1",
        plugin_name="quarantine_transform",
        node_type=NodeType.TRANSFORM,
    )

    class QuarantineTransform(BaseTransform):
        name = "quarantine_transform"
        input_schema = DynamicSchema
        output_schema = DynamicSchema
        determinism = Determinism.DETERMINISTIC
        plugin_version = "1.0.0"
        _on_error = "discard"  # Errors are discarded (quarantined)

        def __init__(self, node_id: str):
            self.node_id = node_id

        def process(self, row, ctx):
            # Simulate data quality failure
            if row.get("quality") == "bad":
                return TransformResult.error({"reason": "bad_quality"})
            return TransformResult.success(row)

    transform = QuarantineTransform(transform_node.node_id)

    processor = RowProcessor(
        recorder=recorder,
        span_factory=span_factory,
        run_id=run.run_id,
        source_node_id=source_node.node_id,
    )

    ctx = PluginContext(run_id=run.run_id, config={})

    # Process a "bad quality" row - should be quarantined
    results = processor.process_row(
        row_index=0,
        row_data={"value": 42, "quality": "bad"},
        transforms=[transform],
        ctx=ctx,
    )

    assert len(results) == 1
    assert results[0].outcome == RowOutcome.QUARANTINED


def test_transform_error_with_sink_returns_routed(
    self, db: LandscapeDB, run: Run
) -> None:
    """Transform error with on_error=sink_name should return ROUTED."""
    from elspeth.contracts import Determinism, NodeType, RowOutcome
    from elspeth.plugins.base import BaseTransform
    from elspeth.plugins.results import TransformResult
    from elspeth.plugins.schemas import DynamicSchema

    recorder = LandscapeRecorder(db)
    span_factory = SpanFactory()

    source_node = recorder.register_node(
        run_id=run.run_id,
        node_id="source_1",
        plugin_name="test_source",
        node_type=NodeType.SOURCE,
    )
    transform_node = recorder.register_node(
        run_id=run.run_id,
        node_id="transform_1",
        plugin_name="error_routing_transform",
        node_type=NodeType.TRANSFORM,
    )

    class ErrorRoutingTransform(BaseTransform):
        name = "error_routing_transform"
        input_schema = DynamicSchema
        output_schema = DynamicSchema
        determinism = Determinism.DETERMINISTIC
        plugin_version = "1.0.0"
        _on_error = "error_sink"  # Errors go to error_sink

        def __init__(self, node_id: str):
            self.node_id = node_id

        def process(self, row, ctx):
            if row.get("quality") == "bad":
                return TransformResult.error({"reason": "bad_quality"})
            return TransformResult.success(row)

    transform = ErrorRoutingTransform(transform_node.node_id)

    processor = RowProcessor(
        recorder=recorder,
        span_factory=span_factory,
        run_id=run.run_id,
        source_node_id=source_node.node_id,
    )

    ctx = PluginContext(run_id=run.run_id, config={})

    # Process a "bad quality" row - should be routed to error_sink
    results = processor.process_row(
        row_index=0,
        row_data={"value": 42, "quality": "bad"},
        transforms=[transform],
        ctx=ctx,
    )

    assert len(results) == 1
    assert results[0].outcome == RowOutcome.ROUTED
    assert results[0].sink_name == "error_sink"
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/engine/test_processor.py::test_transform_error_with_discard_returns_quarantined -v
pytest tests/engine/test_processor.py::test_transform_error_with_sink_returns_routed -v
```

Expected: FAIL (returns FAILED instead of QUARANTINED/ROUTED)

### Step 3: Update processor to handle error routing

```python
# src/elspeth/engine/processor.py - update _process_single_token method
# Find the BaseTransform handling section (around line 259-276) and replace:

            elif isinstance(transform, BaseTransform):
                # Regular transform
                result, current_token, error_sink = self._transform_executor.execute_transform(
                    transform=transform,
                    token=current_token,
                    ctx=ctx,
                    step_in_pipeline=step,
                )

                if result.status == "error":
                    # Determine outcome based on error routing
                    if error_sink == "discard":
                        # Intentionally discarded - QUARANTINED
                        return (
                            RowResult(
                                token=current_token,
                                final_data=current_token.row_data,
                                outcome=RowOutcome.QUARANTINED,
                            ),
                            child_items,
                        )
                    else:
                        # Routed to error sink
                        return (
                            RowResult(
                                token=current_token,
                                final_data=current_token.row_data,
                                outcome=RowOutcome.ROUTED,
                                sink_name=error_sink,
                            ),
                            child_items,
                        )
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/engine/test_processor.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor.py
git commit -m "feat(processor): return QUARANTINED/ROUTED for transform errors (WP-10 Task 2)

- on_error='discard' → RowOutcome.QUARANTINED
- on_error=sink_name → RowOutcome.ROUTED with sink_name
- Maintains backwards compatibility with existing error handling

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update Existing Tests for New Return Signature

**Files:**
- Modify: `tests/engine/test_executors.py`
- Modify: Any other files that call `execute_transform()`

### Step 1: Find all callers of execute_transform

```bash
grep -r "execute_transform" tests/
```

### Step 2: Update each test to unpack three values

Pattern change:
```python
# Old:
result, updated_token = executor.execute_transform(...)

# New:
result, updated_token, error_sink = executor.execute_transform(...)
# Or if error_sink not needed:
result, updated_token, _ = executor.execute_transform(...)
```

### Step 3: Run all executor tests

```bash
pytest tests/engine/test_executors.py -v
```

Expected: PASS

### Step 4: Run all processor tests

```bash
pytest tests/engine/test_processor.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add tests/
git commit -m "test: update tests for execute_transform 3-tuple return (WP-10 Task 3)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Update Orchestrator Metrics for QUARANTINED

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`
- Test: `tests/engine/test_orchestrator.py`

### Step 1: Write test for orchestrator counting quarantined rows

```python
# tests/engine/test_orchestrator.py - add to appropriate test class

def test_orchestrator_counts_quarantined_rows(self) -> None:
    """Orchestrator should count QUARANTINED rows separately."""
    from elspeth.contracts import Determinism, PluginSchema
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from elspeth.plugins.base import BaseTransform
    from elspeth.plugins.results import TransformResult
    from elspeth.engine.artifacts import ArtifactDescriptor

    db = LandscapeDB.in_memory()

    class RowSchema(PluginSchema):
        value: int
        quality: str

    class ListSource(_TestSourceBase):
        name = "list_source"
        output_schema = RowSchema

        def load(self, ctx):
            yield {"value": 1, "quality": "good"}
            yield {"value": 2, "quality": "bad"}  # Will be quarantined
            yield {"value": 3, "quality": "good"}

        def close(self):
            pass

    class QualityTransform(BaseTransform):
        name = "quality_transform"
        input_schema = RowSchema
        output_schema = RowSchema
        determinism = Determinism.DETERMINISTIC
        plugin_version = "1.0.0"
        _on_error = "discard"  # Quarantine bad quality
        node_id: str | None = None

        def process(self, row, ctx):
            if row.get("quality") == "bad":
                return TransformResult.error({"reason": "bad_quality"})
            return TransformResult.success(row)

    class CollectSink(_TestSinkBase):
        name = "collect_sink"
        results: list = []
        config: dict = {}

        def write(self, rows, ctx):
            self.results.extend(rows)
            import hashlib
            content = str(rows).encode()
            return ArtifactDescriptor(
                artifact_type="file",
                path_or_uri="memory://test",
                content_hash=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
            )

        def close(self):
            pass

    source = ListSource()
    transform = QualityTransform()
    sink = CollectSink()

    config = PipelineConfig(
        source=source,
        transforms=[transform],
        sinks={"default": sink},
    )

    graph = _build_test_graph(config)
    orchestrator = Orchestrator(db)
    run_result = orchestrator.run(config, graph=graph)

    assert run_result.status == "completed"
    assert run_result.rows_processed == 3
    assert run_result.rows_succeeded == 2  # good quality rows
    assert run_result.rows_quarantined == 1  # bad quality row
    # Only good quality rows written to sink
    assert len(sink.results) == 2
```

### Step 2: Add rows_quarantined to RunResult and orchestrator

```python
# src/elspeth/engine/orchestrator.py - update RunResult dataclass and counting logic

@dataclass
class RunResult:
    """Result of a pipeline run."""
    status: str
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed: int
    rows_quarantined: int = 0  # Add this field

# In _execute_run method, add counting for QUARANTINED:
                    elif result.outcome == RowOutcome.QUARANTINED:
                        rows_quarantined += 1
```

### Step 3: Run test

```bash
pytest tests/engine/test_orchestrator.py::test_orchestrator_counts_quarantined_rows -v
```

Expected: PASS

### Step 4: Commit

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "feat(orchestrator): add rows_quarantined metric (WP-10 Task 4)

- Add rows_quarantined field to RunResult
- Count QUARANTINED outcomes separately from FAILED
- Pipeline continues after quarantine (doesn't crash)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Integration Test - Full Quarantine Flow

**Files:**
- Test: `tests/engine/test_processor.py`

### Step 1: Write integration test

```python
# tests/engine/test_processor.py - add new test class

class TestQuarantineIntegration:
    """Integration tests for quarantine flow."""

    def test_pipeline_continues_after_quarantine(
        self, db: LandscapeDB, run: Run
    ) -> None:
        """Pipeline should continue processing after quarantining a row."""
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import DynamicSchema

        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()

        source_node = recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            node_id="transform_1",
            plugin_name="quality_gate",
            node_type=NodeType.TRANSFORM,
        )

        class QualityGate(BaseTransform):
            name = "quality_gate"
            input_schema = DynamicSchema
            output_schema = DynamicSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"
            _on_error = "discard"

            def __init__(self, node_id: str):
                self.node_id = node_id

            def process(self, row, ctx):
                # Quarantine rows with negative values
                if row.get("value", 0) < 0:
                    return TransformResult.error({"reason": "negative_value"})
                return TransformResult.success({**row, "validated": True})

        transform = QualityGate(transform_node.node_id)

        processor = RowProcessor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run.run_id,
            source_node_id=source_node.node_id,
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process multiple rows - some good, some bad
        test_rows = [
            {"value": 10},   # Good
            {"value": -5},   # Quarantined
            {"value": 20},   # Good
            {"value": -1},   # Quarantined
            {"value": 30},   # Good
        ]

        all_results = []
        for i, row_data in enumerate(test_rows):
            results = processor.process_row(
                row_index=i,
                row_data=row_data,
                transforms=[transform],
                ctx=ctx,
            )
            all_results.extend(results)

        # Should have 5 results total
        assert len(all_results) == 5

        # Count outcomes
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]
        quarantined = [r for r in all_results if r.outcome == RowOutcome.QUARANTINED]

        assert len(completed) == 3  # Positive values
        assert len(quarantined) == 2  # Negative values

        # Completed rows should have validated flag
        for result in completed:
            assert result.final_data.get("validated") is True

        # Quarantined rows should have original data (not modified)
        for result in quarantined:
            assert result.final_data.get("value") < 0
            assert "validated" not in result.final_data


    def test_quarantine_records_audit_trail(
        self, db: LandscapeDB, run: Run
    ) -> None:
        """Quarantined rows should be recorded in audit trail."""
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import DynamicSchema

        recorder = LandscapeRecorder(db)
        span_factory = SpanFactory()

        source_node = recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
        )
        transform_node = recorder.register_node(
            run_id=run.run_id,
            node_id="transform_1",
            plugin_name="quality_gate",
            node_type=NodeType.TRANSFORM,
        )

        class QualityGate(BaseTransform):
            name = "quality_gate"
            input_schema = DynamicSchema
            output_schema = DynamicSchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"
            _on_error = "discard"

            def __init__(self, node_id: str):
                self.node_id = node_id

            def process(self, row, ctx):
                if row.get("invalid"):
                    return TransformResult.error({"reason": "invalid_row"})
                return TransformResult.success(row)

        transform = QualityGate(transform_node.node_id)

        processor = RowProcessor(
            recorder=recorder,
            span_factory=span_factory,
            run_id=run.run_id,
            source_node_id=source_node.node_id,
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process an invalid row
        results = processor.process_row(
            row_index=0,
            row_data={"value": 42, "invalid": True},
            transforms=[transform],
            ctx=ctx,
        )

        assert len(results) == 1
        assert results[0].outcome == RowOutcome.QUARANTINED

        # Verify node_state was recorded with failed status
        # (The error is recorded in node_states, ctx.record_transform_error records to transform_errors)
        from sqlalchemy import select
        from elspeth.core.landscape.schema import node_states_table

        with db.connection() as conn:
            states = conn.execute(
                select(node_states_table).where(
                    node_states_table.c.node_id == transform_node.node_id
                )
            ).fetchall()

        assert len(states) == 1
        assert states[0].status == "failed"  # Transform recorded as failed
```

### Step 2: Run tests

```bash
pytest tests/engine/test_processor.py::TestQuarantineIntegration -v
```

Expected: PASS

### Step 3: Commit

```bash
git add tests/engine/test_processor.py
git commit -m "test(processor): add quarantine integration tests (WP-10 Task 5)

- Test pipeline continues after quarantine
- Test multiple rows with mixed outcomes
- Test audit trail records quarantined rows

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update Tracker

**Files:**
- Modify: `docs/plans/2026-01-17-plugin-refactor-tracker.md`

### Step 1: Mark WP-10 complete

Update tracker with completion status and verification results.

### Step 2: Commit

```bash
git add docs/plans/2026-01-17-plugin-refactor-tracker.md
git commit -m "docs(tracker): mark WP-10 complete

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification Checklist

After all tasks complete, verify:

- [ ] `TransformExecutor.execute_transform()` returns 3-tuple: `(result, token, error_sink)`
- [ ] `error_sink` is None for success, "discard" for quarantine, sink_name for routing
- [ ] Processor returns `QUARANTINED` when `error_sink == "discard"`
- [ ] Processor returns `ROUTED` with sink_name when `error_sink` is a sink name
- [ ] Orchestrator counts `rows_quarantined` separately
- [ ] Pipeline continues processing after quarantine (doesn't crash)
- [ ] Audit trail records quarantined rows (node_state with status="failed")
- [ ] All existing tests pass after signature update
- [ ] `mypy --strict` passes
- [ ] All tests pass

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking change to return signature | Update all callers in same commit |
| Test failures from unpacking | Search for all `execute_transform` calls |
| Confusion FAILED vs QUARANTINED | Document: FAILED=unexpected, QUARANTINED=intentional |

---

## Notes

1. **Source-level quarantine already works**: Sources use `ctx.record_validation_error()` and don't yield invalid rows. This is handled at ingestion, not in processor.

2. **FAILED is still possible**: If a transform raises an exception (bug), the executor re-raises and the pipeline crashes. FAILED is for unexpected errors. QUARANTINED is for intentional data rejection.

3. **Error routing via ctx**: The executor still calls `ctx.route_to_sink()` for error routing. The processor uses `error_sink` to determine the correct `RowOutcome` but doesn't duplicate the routing logic.

4. **Audit completeness**: Quarantined rows are recorded in:
   - `node_states` with status="failed"
   - `transform_errors` table via `ctx.record_transform_error()`
   - The `RowResult.outcome = QUARANTINED` indicates terminal state

5. **rows_quarantined metric**: Added to `RunResult` so operators can see how many rows were intentionally rejected vs how many succeeded.
