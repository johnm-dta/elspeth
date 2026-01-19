# WP-14c: Aggregation Test Rewrites Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete test coverage for config-driven aggregation triggers (WP-06), focusing on output modes, orchestrator integration, and audit trail verification.

**Architecture:** WP-06 moved trigger evaluation from plugins to the engine. `TriggerConfig` supports count, timeout, and condition triggers with OR logic (first to fire wins). `AggregationSettings` adds `output_mode: single | passthrough | transform` for controlling batch output. Tests verify these behaviors through `RowProcessor` and `Orchestrator`.

**Tech Stack:** pytest, in-memory LandscapeDB, mock aggregation plugins

---

## Current State Analysis

### Existing Coverage ✅

| File | Class | Coverage |
|------|-------|----------|
| `test_triggers.py` | `TestTriggerEvaluator` | count, timeout, condition, combined, reset, which_triggered |
| `test_executors.py` | `TestAggregationExecutor` | batch creation, accept, flush audit, batch members |
| `test_executors.py` | `TestAggregationExecutorTriggers` | trigger evaluation, reset after flush |
| `test_processor.py` | `TestRowProcessorAggregation` | Basic aggregation flow |
| `test_processor.py` | `TestProcessorAggregationTriggers` | Config-driven triggers, count trigger flush |

**Total: ~600+ lines of aggregation tests already exist.**

### Gaps to Fill 🔴

1. **output_mode tests** - `single`, `passthrough`, `transform` not fully exercised
2. **end_of_source trigger** - Implicit trigger at source exhaustion
3. **Orchestrator-level integration** - Full pipeline with aggregation → sink
4. **Multiple aggregations** - Two aggregations in same pipeline
5. **Aggregation + gate interaction** - Routed rows hitting aggregation
6. **Audit trail for CONSUMED_IN_BATCH** - Verify explain() for aggregated tokens
7. **Timeout trigger in real pipeline** - Not just TriggerEvaluator unit test

---

## Task 1: Test output_mode: single

**Files:**
- Test: `tests/engine/test_integration.py`

**Context:** `output_mode: single` means batch produces one aggregated result. This is the default and most common mode for things like sum, average, count aggregations.

**Step 1: Write the failing test**

```python
# Add new test class to tests/engine/test_integration.py

class TestAggregationIntegration:
    """Integration tests for aggregation through full pipeline."""

    def test_aggregation_output_mode_single(self) -> None:
        """output_mode=single: batch produces one aggregated result.

        Pipeline: source (3 rows) → aggregation (count=3) → sink
        Result: sink receives 1 row with aggregated data
        """
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Any:
                yield from self._data

            def close(self) -> None:
                pass

        class SumAggregation(BaseAggregation):
            """Aggregation that sums values."""
            name = "sum_agg"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def __init__(self) -> None:
                super().__init__({})
                self._values: list[int] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                total = sum(self._values)
                self._values = []
                return [{"value": total, "count": len(self._values) or 1}]

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory", size_bytes=0, content_hash=""
                )

            def close(self) -> None:
                pass

        source = ListSource([{"value": 10}, {"value": 20}, {"value": 30}])
        aggregation = SumAggregation()
        sink = CollectSink()

        # Configure: flush after 3 rows, output_mode=single
        agg_settings = AggregationSettings(
            name="sum_agg",
            plugin="sum_agg",
            trigger=TriggerConfig(count=3),
            output_mode="single",
        )

        config = PipelineConfig(
            source=source,
            transforms=[aggregation],
            sinks={"default": sink},
            aggregation_settings={"sum_agg": agg_settings},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config)

        assert result.status == "completed"
        assert result.rows_processed == 3

        # Sink should receive ONE row with sum
        assert len(sink.results) == 1
        assert sink.results[0]["value"] == 60  # 10 + 20 + 30
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_integration.py::TestAggregationIntegration::test_aggregation_output_mode_single -v`
Expected: FAIL - PipelineConfig may not support aggregation_settings directly

**Step 3: Implementation (if needed)**

May need to wire aggregation_settings through PipelineConfig to Orchestrator.

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add tests/engine/test_integration.py
git commit -m "$(cat <<'EOF'
test(aggregation): add output_mode=single integration test

Verifies aggregation with output_mode=single produces one aggregated
result row from batch of inputs.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Test output_mode: passthrough

**Files:**
- Test: `tests/engine/test_integration.py`

**Context:** `output_mode: passthrough` releases all accepted rows unchanged. Useful for rate limiting or grouping without transformation.

**Step 1: Write the failing test**

```python
# Add to TestAggregationIntegration in tests/engine/test_integration.py

def test_aggregation_output_mode_passthrough(self) -> None:
    """output_mode=passthrough: batch releases all rows unchanged.

    Pipeline: source (5 rows) → aggregation (count=3) → sink
    Result: sink receives all 5 rows (3 from first batch, 2 from end_of_source)
    """
    from elspeth.core.config import AggregationSettings, TriggerConfig
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.artifacts import ArtifactDescriptor
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from elspeth.plugins.base import BaseAggregation
    from elspeth.plugins.context import PluginContext
    from elspeth.plugins.results import AcceptResult

    db = LandscapeDB.in_memory()

    class BufferAggregation(BaseAggregation):
        """Aggregation that buffers and releases rows unchanged."""
        name = "buffer_agg"
        input_schema = None
        output_schema = None
        plugin_version = "1.0.0"

        def __init__(self) -> None:
            super().__init__({})
            self._buffer: list[dict[str, Any]] = []

        def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
            self._buffer.append(row)
            return AcceptResult(accepted=True)

        def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
            # Passthrough mode: return all buffered rows
            result = list(self._buffer)
            self._buffer = []
            return result

    # ... similar setup ...

    # Configure: flush after 3 rows, output_mode=passthrough
    agg_settings = AggregationSettings(
        name="buffer_agg",
        plugin="buffer_agg",
        trigger=TriggerConfig(count=3),
        output_mode="passthrough",
    )

    # ... run pipeline with 5 rows ...

    # Sink should receive all 5 rows unchanged
    # First batch: 3 rows, end_of_source flush: 2 remaining rows
    assert len(sink.results) == 5
    assert [r["value"] for r in sink.results] == [1, 2, 3, 4, 5]
```

**Step 2-5: Standard TDD flow**

---

## Task 3: Test end_of_source Implicit Trigger

**Files:**
- Test: `tests/engine/test_integration.py`

**Context:** Per `TriggerConfig` docstring: "end_of_source is IMPLICIT - always checked at source exhaustion."

**Step 1: Write the failing test**

```python
# Add to TestAggregationIntegration

def test_aggregation_flushes_on_source_exhaustion(self) -> None:
    """Aggregation flushes remaining rows when source exhausts.

    Pipeline: source (5 rows) → aggregation (count=100) → sink
    - count=100 never triggers (only 5 rows)
    - end_of_source implicit trigger flushes the 5 accumulated rows
    """
    from elspeth.core.config import AggregationSettings, TriggerConfig
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    db = LandscapeDB.in_memory()

    # ... setup with count=100 trigger that won't fire ...

    source = ListSource([{"value": i} for i in range(5)])  # Only 5 rows
    aggregation = SumAggregation()
    sink = CollectSink()

    agg_settings = AggregationSettings(
        name="sum_agg",
        plugin="sum_agg",
        trigger=TriggerConfig(count=100),  # Won't trigger - only 5 rows
        output_mode="single",
    )

    config = PipelineConfig(
        source=source,
        transforms=[aggregation],
        sinks={"default": sink},
        aggregation_settings={"sum_agg": agg_settings},
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config)

    assert result.status == "completed"

    # Even though count=100 never fired, end_of_source flushed
    assert len(sink.results) == 1
    assert sink.results[0]["value"] == 10  # 0+1+2+3+4
```

**Step 2-5: Standard TDD flow**

---

## Task 4: Test Timeout Trigger in Pipeline

**Files:**
- Test: `tests/engine/test_integration.py`

**Context:** Timeout triggers exist in `TriggerEvaluator` unit tests, but need integration test through actual pipeline.

**Step 1: Write the failing test**

```python
# Add to TestAggregationIntegration

def test_aggregation_timeout_trigger(self) -> None:
    """Timeout trigger fires after elapsed time.

    Pipeline: source → slow_transform → aggregation (timeout=0.05s) → sink
    - Slow transform adds delay between rows
    - Timeout fires before count is reached
    """
    import time

    from elspeth.contracts import TransformResult
    from elspeth.core.config import AggregationSettings, TriggerConfig
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from elspeth.plugins.base import BaseTransform

    db = LandscapeDB.in_memory()

    class SlowTransform(BaseTransform):
        """Transform that adds delay to simulate slow processing."""
        name = "slow_transform"
        input_schema = None
        output_schema = None
        plugin_version = "1.0.0"

        def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
            time.sleep(0.02)  # 20ms delay per row
            return TransformResult.success(row)

    # ... setup ...

    # Configure: timeout=0.05s (50ms), count=1000 (won't trigger)
    agg_settings = AggregationSettings(
        name="buffer_agg",
        plugin="buffer_agg",
        trigger=TriggerConfig(count=1000, timeout_seconds=0.05),
        output_mode="passthrough",
    )

    # Source with 10 rows, each takes 20ms → 200ms total
    # Timeout at 50ms should fire after ~2-3 rows
    source = ListSource([{"value": i} for i in range(10)])

    # ... run pipeline ...

    # Timeout should have fired multiple times
    # With 20ms per row and 50ms timeout, expect ~2-3 rows per batch
    assert result.status == "completed"
    # Verify which_triggered was "timeout" (if we record this)
```

**Step 2-5: Standard TDD flow**

---

## Task 5: Test Multiple Aggregations in Pipeline

**Files:**
- Test: `tests/engine/test_integration.py`

**Step 1: Write the failing test**

```python
# Add to TestAggregationIntegration

def test_multiple_aggregations_in_pipeline(self) -> None:
    """Pipeline with two sequential aggregations.

    Pipeline: source → agg1 (count=2) → agg2 (count=3) → sink

    - agg1 groups by 2: [1,2] → sum=3, [3,4] → sum=7, [5] → sum=5
    - agg2 groups by 3: [3,7,5] → sum=15
    """
    from elspeth.core.config import AggregationSettings, TriggerConfig
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    db = LandscapeDB.in_memory()

    # ... create two sum aggregations ...

    source = ListSource([{"value": i} for i in range(1, 6)])  # 1,2,3,4,5
    agg1 = SumAggregation()
    agg1.name = "sum_agg_1"
    agg2 = SumAggregation()
    agg2.name = "sum_agg_2"
    sink = CollectSink()

    agg_settings = {
        "sum_agg_1": AggregationSettings(
            name="sum_agg_1",
            plugin="sum_agg",
            trigger=TriggerConfig(count=2),
            output_mode="single",
        ),
        "sum_agg_2": AggregationSettings(
            name="sum_agg_2",
            plugin="sum_agg",
            trigger=TriggerConfig(count=3),
            output_mode="single",
        ),
    }

    config = PipelineConfig(
        source=source,
        transforms=[agg1, agg2],
        sinks={"default": sink},
        aggregation_settings=agg_settings,
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config)

    assert result.status == "completed"
    # Final aggregated value depends on timing of flushes
    # agg1 produces 3 rows: 3, 7, 5
    # agg2 produces 1 row: 15
    assert len(sink.results) == 1
    assert sink.results[0]["value"] == 15
```

**Step 2-5: Standard TDD flow**

---

## Task 6: Test Aggregation with Gate Routing

**Files:**
- Test: `tests/engine/test_integration.py`

**Step 1: Write the failing test**

```python
# Add to TestAggregationIntegration

def test_aggregation_after_gate_routing(self) -> None:
    """Aggregation receives rows after gate routing.

    Pipeline: source → gate (route high/low) → aggregation → sinks

    High-value rows (>50) aggregate separately from low-value rows.
    """
    from elspeth.core.config import AggregationSettings, GateSettings, TriggerConfig
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    db = LandscapeDB.in_memory()

    # ... setup ...

    source = ListSource([
        {"value": 10}, {"value": 20},   # Low path
        {"value": 60}, {"value": 70},   # High path
    ])

    gate = GateSettings(
        name="value_router",
        condition="row['value'] > 50",
        routes={"true": "high_agg", "false": "low_agg"},
    )

    # Two aggregations, one per path
    high_agg = SumAggregation()
    high_agg.name = "high_agg"
    low_agg = SumAggregation()
    low_agg.name = "low_agg"

    agg_settings = {
        "high_agg": AggregationSettings(
            name="high_agg",
            plugin="sum_agg",
            trigger=TriggerConfig(count=2),
            output_mode="single",
        ),
        "low_agg": AggregationSettings(
            name="low_agg",
            plugin="sum_agg",
            trigger=TriggerConfig(count=2),
            output_mode="single",
        ),
    }

    high_sink = CollectSink()
    low_sink = CollectSink()

    # ... complex graph setup with gate → aggregation → sink paths ...

    result = orchestrator.run(config, graph=graph)

    assert result.status == "completed"
    # High path: 60 + 70 = 130
    assert high_sink.results[0]["value"] == 130
    # Low path: 10 + 20 = 30
    assert low_sink.results[0]["value"] == 30
```

**Step 2-5: Standard TDD flow**

---

## Task 7: Test Audit Trail for CONSUMED_IN_BATCH

**Files:**
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

```python
# Add to TestProcessorAggregationTriggers in tests/engine/test_processor.py

def test_aggregated_tokens_audit_trail(self) -> None:
    """Tokens consumed in batch have complete audit trail.

    For any row that was aggregated, explain() should show:
    - Source row entry
    - Aggregation accept (CONSUMED_IN_BATCH)
    - Link to batch that consumed it
    - Batch output row
    """
    from elspeth.core.config import AggregationSettings, TriggerConfig
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from elspeth.plugins.results import RowOutcome

    db = LandscapeDB.in_memory()

    # ... setup pipeline with aggregation ...

    result = orchestrator.run(config)

    # Query audit trail for original tokens
    recorder = LandscapeRecorder(db)

    # Get all tokens from this run
    with db._engine.connect() as conn:
        from sqlalchemy import text

        # Find batch members
        batch_members = conn.execute(
            text("""
                SELECT bm.token_id, bm.batch_id, bm.ordinal
                FROM batch_members bm
                JOIN batches b ON bm.batch_id = b.batch_id
                WHERE b.run_id = :run_id
            """),
            {"run_id": result.run_id},
        ).fetchall()

        assert len(batch_members) == 3, "Should have 3 tokens in batch"

        # Verify each token has node_state showing CONSUMED_IN_BATCH
        for token_id, batch_id, ordinal in batch_members:
            states = recorder.get_node_states_for_token(token_id)

            # Find aggregation state
            agg_states = [s for s in states if s.node_type == "aggregation"]
            assert len(agg_states) == 1
            assert agg_states[0].status.value == "consumed"  # Or appropriate status

            # Verify batch_id is recorded in metadata
            metadata = agg_states[0].metadata or {}
            assert metadata.get("batch_id") == batch_id
```

**Step 2-5: Standard TDD flow**

---

## Task 8: Test Condition Trigger with Expression

**Files:**
- Test: `tests/engine/test_integration.py`

**Context:** Condition triggers use `ExpressionParser` with special variables `row['batch_count']` and `row['batch_age_seconds']`.

**Step 1: Write the failing test**

```python
# Add to TestAggregationIntegration

def test_aggregation_condition_trigger(self) -> None:
    """Condition trigger fires when expression is true.

    Condition: batch_count >= 3 AND batch_age_seconds > 0.01
    This tests the special variables available in condition expressions.
    """
    import time

    from elspeth.core.config import AggregationSettings, TriggerConfig
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    db = LandscapeDB.in_memory()

    # ... setup ...

    # Condition: need both count and time
    agg_settings = AggregationSettings(
        name="buffer_agg",
        plugin="buffer_agg",
        trigger=TriggerConfig(
            condition="row['batch_count'] >= 3 and row['batch_age_seconds'] > 0.01"
        ),
        output_mode="passthrough",
    )

    source = ListSource([{"value": i} for i in range(10)])

    # Add small delay to ensure time passes
    class DelayTransform(BaseTransform):
        def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
            time.sleep(0.005)  # 5ms
            return TransformResult.success(row)

    # ... run pipeline ...

    # Should flush after 3 rows when age > 10ms
    assert result.status == "completed"
```

**Step 2-5: Standard TDD flow**

---

## Task 9: Verify Tests Pass and Coverage

**Step 1: Run all aggregation tests**

```bash
pytest tests/engine/test_triggers.py -v
pytest tests/engine/test_executors.py::TestAggregationExecutor -v
pytest tests/engine/test_executors.py::TestAggregationExecutorTriggers -v
pytest tests/engine/test_processor.py::TestRowProcessorAggregation -v
pytest tests/engine/test_processor.py::TestProcessorAggregationTriggers -v
pytest tests/engine/test_integration.py::TestAggregationIntegration -v
```

**Step 2: Verify coverage**

```bash
pytest tests/engine/test_triggers.py \
       tests/engine/test_executors.py \
       tests/engine/test_processor.py::TestRowProcessorAggregation \
       tests/engine/test_processor.py::TestProcessorAggregationTriggers \
       tests/engine/test_integration.py::TestAggregationIntegration \
    --cov=src/elspeth/engine/triggers \
    --cov=src/elspeth/engine/executors \
    --cov-report=term-missing
```

Target: >90% coverage for TriggerEvaluator and AggregationExecutor

**Step 3: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test(wp-14c): complete aggregation test coverage

WP-14c implementation complete:
- output_mode: single, passthrough, transform
- end_of_source implicit trigger
- Timeout trigger in real pipeline
- Multiple aggregations in pipeline
- Aggregation + gate routing interaction
- Audit trail for CONSUMED_IN_BATCH tokens
- Condition trigger with batch_count/batch_age_seconds

Total aggregation test coverage: ~1000+ lines across 4 test files.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Task | Description | Estimated Time |
|------|-------------|----------------|
| 1 | output_mode: single | 30 min |
| 2 | output_mode: passthrough | 30 min |
| 3 | end_of_source implicit trigger | 30 min |
| 4 | Timeout trigger in pipeline | 45 min |
| 5 | Multiple aggregations | 45 min |
| 6 | Aggregation + gate routing | 45 min |
| 7 | Audit trail for CONSUMED_IN_BATCH | 30 min |
| 8 | Condition trigger with expression | 30 min |
| 9 | Coverage verification | 15 min |

**Total estimated time: ~5 hours**

---

## Test Coverage Checklist

- [ ] output_mode: single (one aggregated result)
- [ ] output_mode: passthrough (release all rows unchanged)
- [ ] output_mode: transform (batch applies transform)
- [ ] end_of_source implicit trigger
- [ ] Timeout trigger fires in real pipeline
- [ ] Combined triggers (count + timeout)
- [ ] Multiple aggregations in sequence
- [ ] Aggregation after gate routing
- [ ] Audit trail shows batch membership
- [ ] Condition trigger with batch_count and batch_age_seconds
- [ ] >90% coverage for TriggerEvaluator and AggregationExecutor

---

## Notes on ELSPETH Aggregation Architecture

Per the Three-Tier Trust Model:
- **Aggregation plugins** are system-owned code (Tier 1) - bugs should crash
- **Row data** passed to aggregations is Tier 2 (elevated trust) - wrap operations on values
- **Trigger conditions** use `ExpressionParser` - same security model as gates

The audit trail records:
- Each token's acceptance into a batch (`batch_members` table)
- Batch lifecycle (`batches` table with draft → committed → flushed states)
- Which trigger fired (`which_triggered()` method on TriggerEvaluator)
