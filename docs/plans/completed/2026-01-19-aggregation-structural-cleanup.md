# Aggregation Structural Cleanup Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the transition to fully structural aggregation by removing plugin-level `AggregationProtocol` and making the engine handle all batching internally.

**Context:** WP-06 moved trigger evaluation to the engine but left `AggregationProtocol` with `accept()`/`flush()` methods. This plan completes the cleanup so that:
- Engine buffers rows directly (no plugin `accept()` calls)
- When trigger fires, engine passes `list[dict]` to a regular Transform
- `AggregationProtocol` and `BaseAggregation` are deleted entirely

**Architecture After Cleanup:**

```
BEFORE (Current - Hybrid)
─────────────────────────
Source → Engine → AggregationPlugin.accept(row) → Engine decides trigger → AggregationPlugin.flush()
                  ↑                                                         ↑
                  Plugin buffers rows                                       Plugin processes batch

AFTER (Target - Fully Structural)
─────────────────────────────────
Source → Engine buffers rows → Engine decides trigger → Transform.process(rows: list[dict])
         ↑                                              ↑
         Engine owns the buffer                         Regular transform receives batch
```

**Tech Stack:** Python 3.12, Pydantic, existing TriggerEvaluator from WP-06

**Dependencies:** WP-06 must be complete (TriggerEvaluator exists)

---

## Breaking Change Impact Assessment

**This plan makes breaking API changes:**

| Component | Change | Impact |
|-----------|--------|--------|
| `AggregationProtocol` | DELETED | Any code referencing this protocol breaks |
| `BaseAggregation` | DELETED | Any aggregation plugins break |
| `AggregationExecutor.accept()` | Signature changes | Engine internals only |
| `TransformProtocol.process()` | Now accepts `dict \| list[dict]` | All transforms must handle both |

**Files to update:**

```bash
# Find all references
grep -rn "AggregationProtocol\|BaseAggregation" src/elspeth tests --include="*.py"
grep -rn "\.accept\(" src/elspeth/engine --include="*.py" | grep -i aggreg
```

---

## Scope Discipline

**DO NOT:**
- Add features not in this plan
- Refactor unrelated code
- Add extra abstractions "for future use"
- Change Transform behavior beyond batch support

**DO:**
- Follow TDD exactly as written
- Delete code completely (no deprecation warnings)
- Update all call sites in the same commit

---

## Task 1: Update TransformProtocol to Accept Batched Rows

**Files:**
- Modify: `src/elspeth/plugins/protocols.py`
- Modify: `src/elspeth/plugins/base.py`
- Test: `tests/plugins/test_protocols.py`

**Step 1: Write the failing test**

Add to `tests/plugins/test_protocols.py`:

```python
class TestTransformBatchSupport:
    """Tests for batch-aware transform protocol."""

    def test_transform_process_single_row(self) -> None:
        """Transform.process() accepts single row dict."""
        from elspeth.plugins.base import BaseTransform

        class SingleTransform(BaseTransform):
            name = "single"
            input_schema = AnySchema
            output_schema = AnySchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def process(self, row, ctx):
                return TransformResult.success({"processed": row["value"]})

        transform = SingleTransform({})
        result = transform.process({"value": 1}, mock_ctx)
        assert result.row == {"processed": 1}

    def test_transform_process_batch_rows(self) -> None:
        """Transform.process() accepts list of row dicts."""
        from elspeth.plugins.base import BaseTransform

        class BatchTransform(BaseTransform):
            name = "batch"
            input_schema = AnySchema
            output_schema = AnySchema
            determinism = Determinism.DETERMINISTIC
            plugin_version = "1.0.0"

            def process(self, rows, ctx):
                # When given a list, process as batch
                if isinstance(rows, list):
                    total = sum(r["value"] for r in rows)
                    return TransformResult.success({"total": total, "count": len(rows)})
                # Single row
                return TransformResult.success({"value": rows["value"]})

        transform = BatchTransform({})

        # Batch mode
        result = transform.process([{"value": 1}, {"value": 2}, {"value": 3}], mock_ctx)
        assert result.row == {"total": 6, "count": 3}

    def test_transform_is_batch_aware_property(self) -> None:
        """Transforms declare batch awareness via is_batch_aware property."""
        from elspeth.plugins.base import BaseTransform

        class RegularTransform(BaseTransform):
            name = "regular"
            # ... standard attrs ...

            def process(self, row, ctx):
                return TransformResult.success(row)

        class BatchAwareTransform(BaseTransform):
            name = "batch_aware"
            is_batch_aware = True  # Declares batch support
            # ... standard attrs ...

            def process(self, rows, ctx):
                if isinstance(rows, list):
                    return TransformResult.success({"count": len(rows)})
                return TransformResult.success(rows)

        regular = RegularTransform({})
        batch = BatchAwareTransform({})

        assert getattr(regular, 'is_batch_aware', False) is False
        assert batch.is_batch_aware is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_protocols.py::TestTransformBatchSupport -v`

**Step 3: Update TransformProtocol**

In `src/elspeth/plugins/protocols.py`, update `TransformProtocol`:

```python
@runtime_checkable
class TransformProtocol(Protocol):
    """Protocol for row transforms.

    Transforms process rows and emit results. They can operate in two modes:
    - Single row: process(row: dict, ctx) -> TransformResult
    - Batch: process(rows: list[dict], ctx) -> TransformResult (if is_batch_aware=True)

    The engine decides which mode to use based on:
    - is_batch_aware attribute (default False)
    - Aggregation configuration in pipeline

    For batch-aware transforms used in aggregation nodes:
    - Engine buffers rows until trigger fires
    - Engine calls process(rows: list[dict], ctx)
    - Transform returns single aggregated result or multiple results
    """

    name: str
    input_schema: type["PluginSchema"]
    output_schema: type["PluginSchema"]
    node_id: str | None
    determinism: Determinism
    plugin_version: str
    _on_error: str | None

    # Batch support (for aggregation nodes)
    is_batch_aware: bool  # Default False in BaseTransform

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: "PluginContext",
    ) -> "TransformResult":
        """Process row(s).

        Args:
            row: Single row dict OR list of row dicts (if is_batch_aware=True)
            ctx: Plugin context

        Returns:
            TransformResult with processed output
        """
        ...
```

**Step 4: Update BaseTransform**

In `src/elspeth/plugins/base.py`, update `BaseTransform`:

```python
class BaseTransform(ABC):
    """Base class for transforms.

    Subclass and implement process().

    For batch-aware transforms (used in aggregation nodes):
    - Set is_batch_aware = True
    - process() will receive list[dict] when used in aggregation
    """

    # ... existing attributes ...

    # Batch support - override to True for batch-aware transforms
    is_batch_aware: bool = False

    @abstractmethod
    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: "PluginContext",
    ) -> TransformResult:
        """Process row(s).

        Args:
            row: Single row dict, OR list of row dicts if is_batch_aware=True

        Returns:
            TransformResult with processed output
        """
        ...
```

**Step 5: Run tests**

Run: `pytest tests/plugins/test_protocols.py::TestTransformBatchSupport -v`

**Step 6: Commit**

```bash
git add -A && git commit -m "feat(plugins): add batch support to TransformProtocol

Transforms can now declare is_batch_aware=True to receive list[dict]
when used in aggregation nodes. Engine buffers rows and passes batch
to transform when trigger fires.

Default is_batch_aware=False for backwards compatibility.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Update AggregationExecutor to Buffer Rows Internally

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Test: `tests/engine/test_executors.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_executors.py`:

```python
class TestAggregationExecutorBuffering:
    """Tests for engine-level row buffering in AggregationExecutor."""

    def test_executor_buffers_rows_internally(
        self,
        recorder: "LandscapeRecorder",
        span_factory: "SpanFactory",
        ctx: "PluginContext",
    ) -> None:
        """Executor buffers rows without calling plugin.accept()."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=3),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id="run-1",
            aggregation_settings={"agg-node-1": settings},
        )

        # Buffer 3 rows
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                run_id="run-1",
                row_data={"value": i},
            )
            executor.buffer_row("agg-node-1", token)

        # Check buffer
        buffered = executor.get_buffered_rows("agg-node-1")
        assert len(buffered) == 3
        assert [r["value"] for r in buffered] == [0, 1, 2]

    def test_executor_clears_buffer_after_flush(
        self,
        recorder: "LandscapeRecorder",
        span_factory: "SpanFactory",
        ctx: "PluginContext",
    ) -> None:
        """Executor clears buffer after flush."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id="run-1",
            aggregation_settings={"agg-node-1": settings},
        )

        # Buffer rows
        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                run_id="run-1",
                row_data={"value": i},
            )
            executor.buffer_row("agg-node-1", token)

        # Get buffered rows (this also clears the buffer)
        buffered = executor.flush_buffer("agg-node-1")
        assert len(buffered) == 2

        # Buffer should be empty
        assert executor.get_buffered_rows("agg-node-1") == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_executors.py::TestAggregationExecutorBuffering -v`

**Step 3: Update AggregationExecutor**

In `src/elspeth/engine/executors.py`, update `AggregationExecutor`:

```python
class AggregationExecutor:
    """Executes aggregations with engine-level row buffering.

    The engine owns the row buffer. When trigger fires, buffered rows
    are passed to a batch-aware Transform.

    Lifecycle:
    1. buffer_row() - Add row to buffer, update trigger evaluator
    2. should_flush() - Check if trigger condition met
    3. flush_buffer() - Get buffered rows and clear buffer
    4. (Engine calls transform.process(rows) with the buffer)
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
        *,
        aggregation_settings: dict[str, AggregationSettings] | None = None,
    ) -> None:
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._aggregation_settings = aggregation_settings or {}
        self._trigger_evaluators: dict[str, TriggerEvaluator] = {}

        # Engine-owned row buffers (node_id -> list of row dicts)
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        # Token tracking for audit trail (node_id -> list of TokenInfo)
        self._buffer_tokens: dict[str, list[TokenInfo]] = {}
        # Batch IDs for audit (node_id -> current batch_id)
        self._batch_ids: dict[str, str | None] = {}

        # Create trigger evaluators
        for node_id, settings in self._aggregation_settings.items():
            self._trigger_evaluators[node_id] = TriggerEvaluator(settings.trigger)
            self._buffers[node_id] = []
            self._buffer_tokens[node_id] = []

    def buffer_row(
        self,
        node_id: str,
        token: TokenInfo,
    ) -> None:
        """Buffer a row for aggregation.

        Args:
            node_id: Aggregation node ID
            token: Token with row data to buffer
        """
        if node_id not in self._buffers:
            self._buffers[node_id] = []
            self._buffer_tokens[node_id] = []

        # Create batch on first row if needed
        if self._batch_ids.get(node_id) is None:
            batch_id = self._recorder.create_batch(
                run_id=self._run_id,
                aggregation_node_id=node_id,
            )
            self._batch_ids[node_id] = batch_id

        # Buffer the row
        self._buffers[node_id].append(token.row_data)
        self._buffer_tokens[node_id].append(token)

        # Record batch membership
        self._recorder.add_batch_member(
            batch_id=self._batch_ids[node_id],
            token_id=token.token_id,
            ordinal=len(self._buffers[node_id]) - 1,
        )

        # Update trigger evaluator
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator:
            evaluator.record_accept()

    def get_buffered_rows(self, node_id: str) -> list[dict[str, Any]]:
        """Get currently buffered rows (does not clear buffer).

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered row dicts
        """
        return list(self._buffers.get(node_id, []))

    def get_buffered_tokens(self, node_id: str) -> list[TokenInfo]:
        """Get currently buffered tokens (does not clear buffer).

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered TokenInfo objects
        """
        return list(self._buffer_tokens.get(node_id, []))

    def flush_buffer(self, node_id: str) -> list[dict[str, Any]]:
        """Get buffered rows and clear the buffer.

        Args:
            node_id: Aggregation node ID

        Returns:
            List of buffered row dicts
        """
        rows = self._buffers.get(node_id, [])
        self._buffers[node_id] = []
        self._buffer_tokens[node_id] = []

        # Reset trigger evaluator
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator:
            evaluator.reset()

        # Clear batch ID for next batch
        self._batch_ids[node_id] = None

        return rows

    def should_flush(self, node_id: str) -> bool:
        """Check if aggregation should flush based on trigger config."""
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator is None:
            return False
        return evaluator.should_trigger()

    def get_trigger_type(self, node_id: str) -> TriggerType | None:
        """Get the trigger type that fired."""
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator is None:
            return None
        return evaluator.get_trigger_type()
```

**Step 4: Run tests**

Run: `pytest tests/engine/test_executors.py::TestAggregationExecutorBuffering -v`

**Step 5: Commit**

```bash
git add -A && git commit -m "feat(engine): AggregationExecutor buffers rows internally

Engine now owns the row buffer for aggregations:
- buffer_row() adds row to buffer, updates trigger evaluator
- get_buffered_rows() returns current buffer
- flush_buffer() returns and clears buffer
- No plugin.accept() calls - engine handles everything

Prepares for deletion of AggregationProtocol.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2B: Add Buffer Persistence for Crash Recovery

**Context:** The engine now owns row buffers. If we crash mid-batch, we need to persist and restore those buffers. The checkpoint system already has `aggregation_state: dict[str, Any]` - we use this for engine buffers instead of plugin state.

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Modify: `src/elspeth/engine/orchestrator.py` (checkpoint creation)
- Test: `tests/engine/test_executors.py`
- Test: `tests/integration/test_aggregation_recovery.py`

**Step 1: Write failing tests for serialization**

Add to `tests/engine/test_executors.py`:

```python
class TestAggregationExecutorCheckpoint:
    """Tests for buffer serialization/deserialization for crash recovery."""

    def test_get_checkpoint_state_returns_buffer_contents(
        self,
        recorder: "LandscapeRecorder",
        span_factory: "SpanFactory",
    ) -> None:
        """get_checkpoint_state() returns serializable buffer state."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id="run-1",
            aggregation_settings={"agg-node-1": settings},
        )

        # Buffer some rows
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                run_id="run-1",
                row_data={"value": i},
            )
            executor.buffer_row("agg-node-1", token)

        # Get checkpoint state
        state = executor.get_checkpoint_state()

        assert "agg-node-1" in state
        assert state["agg-node-1"]["rows"] == [{"value": 0}, {"value": 1}, {"value": 2}]
        assert state["agg-node-1"]["token_ids"] == ["token-0", "token-1", "token-2"]
        assert state["agg-node-1"]["batch_id"] is not None

    def test_restore_from_checkpoint_restores_buffers(
        self,
        recorder: "LandscapeRecorder",
        span_factory: "SpanFactory",
    ) -> None:
        """restore_from_checkpoint() restores buffer state."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id="run-1",
            aggregation_settings={"agg-node-1": settings},
        )

        # Simulate checkpoint state from previous run
        checkpoint_state = {
            "agg-node-1": {
                "rows": [{"value": 0}, {"value": 1}, {"value": 2}],
                "token_ids": ["token-0", "token-1", "token-2"],
                "batch_id": "batch-123",
            }
        }

        executor.restore_from_checkpoint(checkpoint_state)

        # Buffer should be restored
        buffered = executor.get_buffered_rows("agg-node-1")
        assert len(buffered) == 3
        assert buffered == [{"value": 0}, {"value": 1}, {"value": 2}]

        # Trigger evaluator should reflect restored count
        assert executor._trigger_evaluators["agg-node-1"].batch_count == 3

    def test_checkpoint_state_is_json_serializable(
        self,
        recorder: "LandscapeRecorder",
        span_factory: "SpanFactory",
    ) -> None:
        """Checkpoint state must be JSON serializable."""
        import json

        from elspeth.contracts.identity import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=span_factory,
            run_id="run-1",
            aggregation_settings={"agg-node-1": settings},
        )

        token = TokenInfo(
            row_id="row-0",
            token_id="token-0",
            run_id="run-1",
            row_data={"value": 42, "name": "test"},
        )
        executor.buffer_row("agg-node-1", token)

        state = executor.get_checkpoint_state()

        # Must not raise
        json_str = json.dumps(state)
        restored = json.loads(json_str)
        assert restored == state
```

**Step 2: Implement get_checkpoint_state() and restore_from_checkpoint()**

Add to `AggregationExecutor` in `src/elspeth/engine/executors.py`:

```python
def get_checkpoint_state(self) -> dict[str, Any]:
    """Get serializable state for checkpointing.

    Returns a dict that can be JSON-serialized and stored in
    checkpoint.aggregation_state_json. On recovery, pass this
    to restore_from_checkpoint().

    Returns:
        Dict mapping node_id -> buffer state
    """
    state: dict[str, Any] = {}
    for node_id in self._buffers:
        if self._buffers[node_id]:  # Only include non-empty buffers
            state[node_id] = {
                "rows": list(self._buffers[node_id]),
                "token_ids": [t.token_id for t in self._buffer_tokens[node_id]],
                "batch_id": self._batch_ids.get(node_id),
            }
    return state

def restore_from_checkpoint(self, state: dict[str, Any]) -> None:
    """Restore buffer state from checkpoint.

    Called during recovery to restore buffers from a previous
    run's checkpoint. Also restores trigger evaluator counts.

    Args:
        state: Dict from get_checkpoint_state() of previous run
    """
    for node_id, node_state in state.items():
        rows = node_state.get("rows", [])
        token_ids = node_state.get("token_ids", [])
        batch_id = node_state.get("batch_id")

        # Restore buffer
        self._buffers[node_id] = list(rows)

        # Restore batch ID
        if batch_id:
            self._batch_ids[node_id] = batch_id

        # Restore trigger evaluator count
        evaluator = self._trigger_evaluators.get(node_id)
        if evaluator:
            for _ in range(len(rows)):
                evaluator.record_accept()

        # Note: We don't restore full TokenInfo objects - only token_ids
        # are needed for audit trail. The actual TokenInfo will be
        # reconstructed if needed from the tokens table.
        self._buffer_tokens[node_id] = []  # Clear - will be rebuilt on next buffer_row
```

**Step 3: Run tests**

Run: `pytest tests/engine/test_executors.py::TestAggregationExecutorCheckpoint -v`

**Step 4: Update Orchestrator checkpoint creation**

In `src/elspeth/engine/orchestrator.py`, update checkpoint creation to use executor state:

Find the checkpoint creation code (likely in `_create_checkpoint` or similar) and change:

```python
# BEFORE (uses plugin state)
aggregation_state = self._get_plugin_aggregation_state()

# AFTER (uses engine buffer state)
aggregation_state = self._aggregation_executor.get_checkpoint_state()
```

**Step 5: Update Orchestrator.resume() to restore buffers**

In `src/elspeth/engine/orchestrator.py`, in the `resume()` method:

```python
# BEFORE (restores plugin state)
if resume_point.aggregation_state:
    for node_id, state in resume_point.aggregation_state.items():
        self._aggregation_executor.restore_state(node_id, state)

# AFTER (restores engine buffers)
if resume_point.aggregation_state:
    self._aggregation_executor.restore_from_checkpoint(resume_point.aggregation_state)
```

**Step 6: Update integration recovery tests**

In `tests/integration/test_aggregation_recovery.py`, update tests to verify:
- Buffered rows are persisted in checkpoint
- On recovery, buffer is restored with correct row count
- Trigger evaluator resumes from correct count

**Step 7: Run integration tests**

Run: `pytest tests/integration/test_aggregation_recovery.py -v`

**Step 8: Commit**

```bash
git add -A && git commit -m "feat(engine): add buffer persistence for crash recovery

AggregationExecutor now supports checkpoint/restore:
- get_checkpoint_state() returns serializable buffer state
- restore_from_checkpoint() restores buffers from checkpoint

Orchestrator uses these for aggregation_state in checkpoints.
Buffer contents survive crashes and can be resumed.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Update RowProcessor to Use Engine Buffering

**Files:**
- Modify: `src/elspeth/engine/processor.py`
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_processor.py`:

```python
class TestProcessorBatchTransforms:
    """Tests for batch-aware transforms in RowProcessor."""

    def test_processor_buffers_rows_for_aggregation_node(
        self,
        recorder: "LandscapeRecorder",
        span_factory: "SpanFactory",
        ctx: "PluginContext",
    ) -> None:
        """Processor buffers rows at aggregation nodes."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.processor import RowProcessor
        from elspeth.plugins.base import BaseTransform

        class SumTransform(BaseTransform):
            name = "sum"
            is_batch_aware = True
            # ... other attrs ...

            def process(self, rows, ctx):
                if isinstance(rows, list):
                    total = sum(r["value"] for r in rows)
                    return TransformResult.success({"total": total})
                return TransformResult.success(rows)

        aggregation_settings = {
            "sum-node": AggregationSettings(
                name="sum_batch",
                plugin="sum",
                trigger=TriggerConfig(count=3),
            ),
        }

        processor = RowProcessor(
            recorder=recorder,
            span_factory=span_factory,
            run_id="run-1",
            source_node_id="source-1",
            aggregation_settings=aggregation_settings,
        )

        transform = SumTransform({})
        transform.node_id = "sum-node"

        # Process 3 rows - should buffer first 2, flush on 3rd
        results = []
        for i in range(3):
            result = processor.process_row(
                row_index=i,
                row_data={"value": i + 1},  # 1, 2, 3
                transforms=[transform],
                ctx=ctx,
            )
            results.append(result)

        # First two rows consumed into batch
        assert results[0].outcome == RowOutcome.CONSUMED_IN_BATCH
        assert results[1].outcome == RowOutcome.CONSUMED_IN_BATCH

        # Third row triggers flush - transform receives [1, 2, 3]
        # Result should have total = 6
        assert results[2].final_data == {"total": 6}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_processor.py::TestProcessorBatchTransforms -v`

**Step 3: Update RowProcessor**

In `src/elspeth/engine/processor.py`, update aggregation handling:

```python
def _process_aggregation_node(
    self,
    transform: BaseTransform,
    current_token: TokenInfo,
    ctx: PluginContext,
    step: int,
) -> tuple[RowResult, list[WorkItem]]:
    """Process a row at an aggregation node.

    Engine buffers rows and calls transform.process(rows) when trigger fires.
    """
    node_id = transform.node_id
    assert node_id is not None

    # Buffer the row
    self._aggregation_executor.buffer_row(node_id, current_token)

    # Check if we should flush
    if self._aggregation_executor.should_flush(node_id):
        # Get buffered rows
        buffered_rows = self._aggregation_executor.flush_buffer(node_id)
        buffered_tokens = self._aggregation_executor.get_buffered_tokens(node_id)

        # Call transform with batch
        result = transform.process(buffered_rows, ctx)

        if result.status == "success":
            # Record batch completion in audit trail
            trigger_type = self._aggregation_executor.get_trigger_type(node_id)
            self._recorder.complete_batch(
                batch_id=...,
                trigger_type=trigger_type,
                output_hash=...,
            )

            return (
                RowResult(
                    token=current_token,
                    final_data=result.row,
                    outcome=RowOutcome.COMPLETED,
                ),
                [],
            )
        else:
            # Handle error
            return (
                RowResult(
                    token=current_token,
                    final_data=None,
                    outcome=RowOutcome.FAILED,
                    error=result.reason,
                ),
                [],
            )

    # Not flushing yet - row consumed into batch
    return (
        RowResult(
            token=current_token,
            final_data=current_token.row_data,
            outcome=RowOutcome.CONSUMED_IN_BATCH,
        ),
        [],
    )
```

**Step 4: Run tests**

Run: `pytest tests/engine/test_processor.py::TestProcessorBatchTransforms -v`

**Step 5: Commit**

```bash
git add -A && git commit -m "feat(engine): RowProcessor uses engine buffering for aggregations

Processor now:
- Buffers rows at aggregation nodes via AggregationExecutor
- Checks should_flush() after each buffer
- Calls transform.process(rows: list[dict]) when trigger fires
- No longer calls plugin.accept()

Aggregation is now fully structural.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Delete AggregationProtocol and BaseAggregation

**Files:**
- Modify: `src/elspeth/plugins/protocols.py` - DELETE AggregationProtocol
- Modify: `src/elspeth/plugins/base.py` - DELETE BaseAggregation
- Modify: `src/elspeth/plugins/__init__.py` - Remove exports
- Modify: All tests referencing these

**Step 1: Find all references**

```bash
grep -rn "AggregationProtocol\|BaseAggregation" src/elspeth tests --include="*.py"
grep -rn "from elspeth.plugins import.*Aggregation" src/elspeth tests --include="*.py"
grep -rn "from elspeth.plugins.base import.*Aggregation" src/elspeth tests --include="*.py"
```

**Step 2: Write test that they don't exist**

Add to `tests/plugins/test_protocols.py`:

```python
def test_aggregation_protocol_deleted() -> None:
    """AggregationProtocol should be deleted (aggregation is structural)."""
    import elspeth.plugins.protocols as protocols

    assert not hasattr(protocols, "AggregationProtocol"), (
        "AggregationProtocol should be deleted"
    )


def test_base_aggregation_deleted() -> None:
    """BaseAggregation should be deleted (aggregation is structural)."""
    import elspeth.plugins.base as base

    assert not hasattr(base, "BaseAggregation"), (
        "BaseAggregation should be deleted"
    )
```

**Step 3: Delete AggregationProtocol from protocols.py**

In `src/elspeth/plugins/protocols.py`:
- DELETE the entire `AggregationProtocol` class (lines ~263-351)
- DELETE `AcceptResult` from imports if no longer used

**Step 4: Delete BaseAggregation from base.py**

In `src/elspeth/plugins/base.py`:
- DELETE the entire `BaseAggregation` class (lines ~164-200+)

**Step 5: Update __init__.py exports**

In `src/elspeth/plugins/__init__.py`:
- Remove `BaseAggregation` from imports
- Remove `BaseAggregation` from `__all__`
- Remove `AggregationProtocol` if exported

**Step 6: Update AggregationExecutor**

In `src/elspeth/engine/executors.py`:
- Remove `AggregationProtocol` import
- Remove `accept()` method that calls plugin
- Remove any references to `aggregation.accept()` or `aggregation.flush()`

**Step 7: Fix all broken tests**

For each test that references `AggregationProtocol` or `BaseAggregation`:
- If testing aggregation behavior → rewrite to test engine buffering + batch transform
- If mocking aggregation → replace with batch-aware transform mock
- DELETE tests that are no longer applicable

**Step 8: Run full test suite**

```bash
pytest tests/ -v
```

**Step 9: Commit**

```bash
git add -A && git commit -m "refactor(plugins): delete AggregationProtocol and BaseAggregation

Aggregation is now fully structural:
- Engine buffers rows internally
- Engine evaluates triggers (WP-06)
- Engine calls batch-aware Transform.process(rows: list[dict])
- No plugin-level aggregation interface

BREAKING: AggregationProtocol and BaseAggregation are deleted.
Use is_batch_aware=True on BaseTransform for batch processing.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Delete AcceptResult from Contracts

**Files:**
- Modify: `src/elspeth/contracts/results.py` - DELETE AcceptResult
- Modify: `src/elspeth/contracts/__init__.py` - Remove export
- Update any remaining references

**Step 1: Find all references**

```bash
grep -rn "AcceptResult" src/elspeth tests --include="*.py"
```

**Step 2: Write test that it doesn't exist**

```python
def test_accept_result_deleted() -> None:
    """AcceptResult should be deleted (no plugin-level aggregation)."""
    import elspeth.contracts.results as results

    assert not hasattr(results, "AcceptResult"), (
        "AcceptResult should be deleted"
    )
```

**Step 3: Delete AcceptResult**

In `src/elspeth/contracts/results.py`:
- DELETE the `AcceptResult` dataclass entirely

**Step 4: Update exports**

In `src/elspeth/contracts/__init__.py`:
- Remove `AcceptResult` from imports and `__all__`

**Step 5: Run tests and commit**

```bash
pytest tests/ -v
git add -A && git commit -m "refactor(contracts): delete AcceptResult

No longer needed - engine buffers rows directly without plugin accept/reject.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Update plugin-protocol.md Documentation

**Files:**
- Modify: `docs/contracts/plugin-protocol.md`

**Changes:**
1. Remove `AggregationProtocol` section entirely
2. Update "System Operations" section to clarify aggregation is purely structural
3. Add documentation for `is_batch_aware` transforms
4. Update examples

**Commit:**

```bash
git add -A && git commit -m "docs: update plugin-protocol.md for structural aggregation

- Remove AggregationProtocol section (deleted)
- Clarify aggregation is engine-level only
- Document is_batch_aware for batch transforms
- Update examples

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Update Phase 6 Plan

**Files:**
- Modify: `docs/plans/2026-01-19-phase6-llm-and-azure.md`

**Changes:**

Update A5 (AzureBatchLLMTransform) to use batch-aware Transform pattern:

```python
# BEFORE (in plan)
class AzureBatchLLMTransform(BaseAggregation):
    def collect(self, row, ctx): ...
    def should_flush(self, ctx): ...
    def flush(self, ctx): ...

# AFTER
class AzureBatchLLMTransform(BaseTransform):
    is_batch_aware = True  # Receives list[dict] from engine

    def process(self, rows: list[dict], ctx: PluginContext) -> TransformResult:
        """Process batch of rows through Azure Batch API."""
        # 1. Render templates for all rows
        # 2. Build JSONL
        # 3. Upload to Azure
        # 4. Submit batch job
        # 5. Poll until complete
        # 6. Parse results
        # 7. Return TransformResult with aggregated output
```

**Commit:**

```bash
git add -A && git commit -m "docs: update Phase 6 plan for batch-aware transforms

AzureBatchLLMTransform now extends BaseTransform with is_batch_aware=True
instead of BaseAggregation. Engine handles buffering and triggering.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Run Full Verification

**Step 1: Verify no stale references**

```bash
# Should return nothing
grep -rn "AggregationProtocol\|BaseAggregation" src/elspeth tests --include="*.py" | grep -v "__pycache__"
grep -rn "AcceptResult" src/elspeth tests --include="*.py" | grep -v "__pycache__" | grep -v "test.*deleted"
grep -rn "\.accept\(" src/elspeth/engine --include="*.py" | grep -i aggreg
```

**Step 2: Run mypy**

```bash
mypy src/elspeth --strict
```

**Step 3: Run full test suite**

```bash
pytest tests/ -v
```

**Step 4: Final commit**

```bash
git add -A && git commit -m "chore: verify aggregation structural cleanup complete

Verification:
- AggregationProtocol deleted
- BaseAggregation deleted
- AcceptResult deleted
- Engine buffers rows internally
- Batch-aware transforms receive list[dict]
- plugin-protocol.md updated
- Phase 6 plan updated
- All tests pass
- No stale references

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Verification Checklist

- [ ] `TransformProtocol.process()` accepts `dict | list[dict]`
- [ ] `BaseTransform.is_batch_aware` attribute exists (default False)
- [ ] `AggregationExecutor.buffer_row()` method exists
- [ ] `AggregationExecutor.flush_buffer()` method exists
- [ ] `AggregationExecutor.get_checkpoint_state()` returns serializable buffer state
- [ ] `AggregationExecutor.restore_from_checkpoint()` restores buffers from checkpoint
- [ ] Checkpoint contains engine buffer state (not plugin state)
- [ ] Recovery restores buffer with correct trigger evaluator count
- [ ] `RowProcessor` uses engine buffering for aggregation nodes
- [ ] `AggregationProtocol` is DELETED
- [ ] `BaseAggregation` is DELETED
- [ ] `AcceptResult` is DELETED
- [ ] No references to deleted types remain
- [ ] `plugin-protocol.md` updated
- [ ] Phase 6 plan updated for batch-aware transforms
- [ ] `mypy --strict` passes
- [ ] All tests pass

---

## Files Changed Summary

| File | Change | Description |
|------|--------|-------------|
| `src/elspeth/plugins/protocols.py` | MODIFY | Add `is_batch_aware`, update `process()` signature, DELETE `AggregationProtocol` |
| `src/elspeth/plugins/base.py` | MODIFY | Add `is_batch_aware` to BaseTransform, DELETE `BaseAggregation` |
| `src/elspeth/plugins/__init__.py` | MODIFY | Remove deleted exports |
| `src/elspeth/contracts/results.py` | MODIFY | DELETE `AcceptResult` |
| `src/elspeth/contracts/__init__.py` | MODIFY | Remove deleted export |
| `src/elspeth/engine/executors.py` | MODIFY | Add buffering methods, checkpoint state, remove plugin calls |
| `src/elspeth/engine/orchestrator.py` | MODIFY | Use executor checkpoint state, restore buffers on resume |
| `src/elspeth/engine/processor.py` | MODIFY | Use engine buffering, call batch transforms |
| `docs/contracts/plugin-protocol.md` | MODIFY | Remove aggregation plugin docs, add batch transform docs |
| `docs/plans/2026-01-19-phase6-llm-and-azure.md` | MODIFY | Update A5 to use batch-aware transform |
| `tests/plugins/test_protocols.py` | MODIFY | Add batch tests, deletion tests |
| `tests/engine/test_executors.py` | MODIFY | Add buffering tests, checkpoint tests, remove accept tests |
| `tests/engine/test_processor.py` | MODIFY | Add batch transform tests |
| `tests/integration/test_aggregation_recovery.py` | MODIFY | Update for engine buffer recovery |

---

## Impact on Phase 6 Plan

After this cleanup, the Phase 6 `AzureBatchLLMTransform` becomes simpler:

```yaml
# Pipeline config
aggregations:
  - node_id: azure_batch_node
    trigger:
      count: 100
      timeout_seconds: 300

transforms:
  - plugin: azure_batch_llm
    node_id: azure_batch_node  # Links to aggregation config
    options:
      model: "gpt-4o"
      template: "Analyze: {{ text }}"
```

The engine:
1. Buffers rows at `azure_batch_node` until count=100 or timeout
2. Calls `AzureBatchLLMTransform.process(rows: list[dict], ctx)`
3. Transform does Azure Batch API call and returns result

No `collect()`, `should_flush()`, `flush()` methods needed. Just `process()`.
