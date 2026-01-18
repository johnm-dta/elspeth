# WP-14d: Integration Test Rewrites Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete end-to-end integration tests that exercise ALL engine features together: fork, coalesce, gates, aggregation, retry, and audit trail verification.

**Architecture:** WP-14a/b/c test individual subsystems in isolation. WP-14d verifies these subsystems work correctly when combined in realistic pipelines. Tests focus on the "audit spine" guarantee: every row reaches a terminal state with complete lineage.

**Tech Stack:** pytest, in-memory LandscapeDB, mock plugins, full Orchestrator

---

## Current State Analysis

### Existing Integration Coverage ✅

| Test Class | Coverage |
|------------|----------|
| `TestEngineIntegration` | Full pipeline, audit spine, routing |
| `TestNoSilentAuditLoss` | Missing edge raises, exceptions propagate |
| `TestAuditTrailCompleteness` | Empty source, multiple sinks |
| `TestForkIntegration` | Fork execution through RowProcessor |
| `TestPluginSystemIntegration` | Plugin manager workflow |

### Gaps for WP-14d 🔴

1. **Combined feature pipeline** - Fork + Gate + Aggregation + Coalesce in ONE test
2. **Retry integration** - End-to-end retry with RetryManager
3. **explain() query verification** - Full lineage query from output back to source
4. **Diamond DAG pattern** - Fork → parallel transforms → coalesce → sink
5. **Error recovery scenarios** - Partial success, quarantine handling
6. **Orchestrator metrics** - rows_forked, rows_coalesced, rows_quarantined

---

## Task 1: Test Diamond DAG (Fork → Transform → Coalesce)

**Files:**
- Test: `tests/engine/test_integration.py`

**Context:** The "diamond" pattern is a common DAG shape: one input forks to parallel paths, then coalesces back to one output. This is the canonical test for fork/coalesce integration.

**Step 1: Write the failing test**

```python
# Add new test class to tests/engine/test_integration.py

class TestComplexDAGIntegration:
    """Integration tests for complex DAG patterns."""

    def test_diamond_dag_fork_transform_coalesce(self) -> None:
        """Diamond DAG: source → fork → [A, B] → coalesce → sink.

        This tests the complete fork/coalesce cycle:
        1. Source emits row
        2. Gate forks to path_a and path_b
        3. Transform A adds sentiment
        4. Transform B adds entities
        5. Coalesce merges A+B results
        6. Sink receives merged row

        Expected terminal states:
        - 1 FORKED (parent token)
        - 2 intermediate (consumed by coalesce)
        - 1 COALESCED (merged token reaches sink)
        """
        from elspeth.contracts import PluginSchema, RowOutcome, RoutingMode
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseGate, BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import (
            GateResult,
            RoutingAction,
            TransformResult,
        )

        db = LandscapeDB.in_memory()

        class TextSchema(PluginSchema):
            text: str

        class EnrichedSchema(PluginSchema):
            text: str
            sentiment: str | None = None
            entities: list[str] | None = None

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = TextSchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def on_start(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Any:
                yield from self._data

            def close(self) -> None:
                pass

        class ForkGate(BaseGate):
            """Gate that forks to sentiment and entity paths."""
            name = "fork_gate"
            input_schema = TextSchema
            output_schema = TextSchema

            def __init__(self) -> None:
                super().__init__({})

            def evaluate(self, row: dict[str, Any], ctx: Any) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.fork_to_paths(["sentiment_path", "entity_path"]),
                )

        class SentimentTransform(BaseTransform):
            """Adds sentiment analysis."""
            name = "sentiment_transform"
            input_schema = TextSchema
            output_schema = EnrichedSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                # Simulate sentiment analysis
                sentiment = "positive" if "good" in row["text"].lower() else "neutral"
                return TransformResult.success({
                    **row,
                    "sentiment": sentiment,
                })

        class EntityTransform(BaseTransform):
            """Adds entity extraction."""
            name = "entity_transform"
            input_schema = TextSchema
            output_schema = EnrichedSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                # Simulate entity extraction
                entities = ["ACME"] if "acme" in row["text"].lower() else []
                return TransformResult.success({
                    **row,
                    "entities": entities,
                })

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(
                    path="memory://output",
                    size_bytes=len(rows),
                    content_hash="diamond_hash",
                )

            def close(self) -> None:
                pass

        # Create pipeline components
        source = ListSource([{"text": "ACME reported good earnings"}])
        fork_gate = ForkGate()
        sentiment_transform = SentimentTransform()
        entity_transform = EntityTransform()
        sink = CollectSink()

        # Configure coalesce
        coalesce_settings = CoalesceSettings(
            name="merger",
            branches=["sentiment_path", "entity_path"],
            policy="require_all",
            merge="union",
        )

        # Build diamond DAG
        # This requires custom graph construction to express:
        # source → fork_gate → [sentiment_transform, entity_transform] → coalesce → sink

        config = PipelineConfig(
            source=source,
            transforms=[fork_gate, sentiment_transform, entity_transform],
            sinks={"default": sink},
            coalesce_settings={"merger": coalesce_settings},
        )

        graph = _build_diamond_graph(config, fork_gate, coalesce_settings)

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"
        assert result.rows_processed == 1

        # Verify merged output
        assert len(sink.results) == 1
        merged_row = sink.results[0]
        assert merged_row["text"] == "ACME reported good earnings"
        assert merged_row["sentiment"] == "positive"
        assert merged_row["entities"] == ["ACME"]

        # Verify metrics
        assert result.rows_forked == 1  # One fork operation
        assert result.rows_coalesced == 1  # One coalesce operation
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_integration.py::TestComplexDAGIntegration::test_diamond_dag_fork_transform_coalesce -v`
Expected: FAIL - Need to implement `_build_diamond_graph` and wire coalesce through orchestrator

**Step 3: Implement helper and wiring**

```python
def _build_diamond_graph(
    config: PipelineConfig,
    fork_gate: BaseGate,
    coalesce_settings: CoalesceSettings,
) -> ExecutionGraph:
    """Build diamond DAG: source → fork → [A, B] → coalesce → sink."""
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()

    # Source
    graph.add_node("source", node_type="source", plugin_name=config.source.name)

    # Fork gate
    graph.add_node("fork_gate", node_type="gate", plugin_name=fork_gate.name)
    graph.add_edge("source", "fork_gate", label="continue", mode=RoutingMode.MOVE)

    # Parallel transforms
    graph.add_node("sentiment_transform", node_type="transform", plugin_name="sentiment_transform")
    graph.add_node("entity_transform", node_type="transform", plugin_name="entity_transform")

    # Fork edges (COPY mode for parallel execution)
    graph.add_edge("fork_gate", "sentiment_transform", label="sentiment_path", mode=RoutingMode.COPY)
    graph.add_edge("fork_gate", "entity_transform", label="entity_path", mode=RoutingMode.COPY)

    # Coalesce node
    graph.add_node("coalesce", node_type="coalesce", plugin_name="merger")
    graph.add_edge("sentiment_transform", "coalesce", label="sentiment_path", mode=RoutingMode.MOVE)
    graph.add_edge("entity_transform", "coalesce", label="entity_path", mode=RoutingMode.MOVE)

    # Sink
    sink_name = next(iter(config.sinks.keys()))
    graph.add_node(f"sink_{sink_name}", node_type="sink", plugin_name=config.sinks[sink_name].name)
    graph.add_edge("coalesce", f"sink_{sink_name}", label="continue", mode=RoutingMode.MOVE)

    # Populate maps
    graph._sink_id_map = {sink_name: f"sink_{sink_name}"}
    graph._transform_id_map = {0: "fork_gate", 1: "sentiment_transform", 2: "entity_transform"}
    graph._coalesce_id_map = {"merger": "coalesce"}
    graph._route_resolution_map = {
        ("fork_gate", "sentiment_path"): "fork",
        ("fork_gate", "entity_path"): "fork",
    }
    graph._output_sink = sink_name

    return graph
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add tests/engine/test_integration.py
git commit -m "$(cat <<'EOF'
test(integration): add diamond DAG fork/coalesce test

Tests the canonical diamond pattern: source → fork → parallel transforms
→ coalesce → sink. Verifies merged output contains fields from both paths.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Test Combined Features Pipeline

**Files:**
- Test: `tests/engine/test_integration.py`

**Step 1: Write the failing test**

```python
# Add to TestComplexDAGIntegration

def test_full_feature_pipeline(self) -> None:
    """Pipeline using ALL features: gate routing + fork + aggregation + coalesce.

    Pipeline structure:
    source (10 rows)
    → config_gate (routes high/low based on value)
        → high path:
            → fork_gate (splits to A and B)
                → transform_A (adds field_a)
                → transform_B (adds field_b)
            → coalesce (merges A+B)
            → aggregation (batches by 2)
            → high_sink
        → low path:
            → transform_C
            → low_sink

    This exercises:
    - Config gate routing (WP-09)
    - Fork execution (WP-07)
    - Coalesce (WP-08)
    - Aggregation triggers (WP-06)
    """
    from elspeth.core.config import (
        AggregationSettings,
        CoalesceSettings,
        GateSettings,
        TriggerConfig,
    )
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    db = LandscapeDB.in_memory()

    # ... extensive setup ...

    source = ListSource([{"value": i} for i in range(10)])

    # Config gate: values >= 5 go high, others go low
    config_gate = GateSettings(
        name="threshold_gate",
        condition="row['value'] >= 5",
        routes={"true": "high_path", "false": "low_path"},
    )

    # Fork in high path
    fork_gate = ForkGate()

    # Aggregation in high path
    agg_settings = AggregationSettings(
        name="high_agg",
        plugin="sum_agg",
        trigger=TriggerConfig(count=2),
        output_mode="single",
    )

    # Coalesce in high path
    coalesce_settings = CoalesceSettings(
        name="high_coalesce",
        branches=["path_a", "path_b"],
        policy="require_all",
        merge="union",
    )

    # ... run pipeline ...

    result = orchestrator.run(config, graph=graph)

    assert result.status == "completed"
    assert result.rows_processed == 10

    # Verify routing: 5 low (0-4), 5 high (5-9)
    # High path: forked, coalesced, aggregated
    # Low path: straight through

    # High sink receives aggregated batches
    # 5 high rows → 5 forked → 5 coalesced → batched by 2 → 2 full batches + 1 partial
    assert len(high_sink.results) == 3  # 2 full batches + 1 end_of_source

    # Low sink receives 5 rows
    assert len(low_sink.results) == 5

    # Verify metrics capture all features
    assert result.rows_routed == 5  # Low path
    assert result.rows_forked == 5  # High path forked
    assert result.rows_coalesced == 5  # High path coalesced
```

**Step 2-5: Standard TDD flow**

---

## Task 3: Test Retry Integration End-to-End

**Files:**
- Test: `tests/engine/test_integration.py`

**Step 1: Write the failing test**

```python
# Add new test class

class TestRetryIntegration:
    """End-to-end retry behavior tests."""

    def test_transient_failure_retries_and_succeeds(self) -> None:
        """Transform that fails twice then succeeds should complete.

        Pipeline: source → flaky_transform (fails 2x, succeeds 3rd) → sink

        Verifies:
        - RetryManager triggers retries
        - Audit trail records all attempts
        - Final success reaches sink
        """
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.engine.retry import RetryConfig, RetryManager

        db = LandscapeDB.in_memory()

        # Track attempt count
        attempt_counts: dict[int, int] = {}

        class FlakyTransform(BaseTransform):
            """Transform that fails first 2 times, succeeds on 3rd."""
            name = "flaky"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                row_idx = row["index"]
                attempt_counts[row_idx] = attempt_counts.get(row_idx, 0) + 1

                if attempt_counts[row_idx] < 3:
                    return TransformResult.error({
                        "message": f"Transient failure {attempt_counts[row_idx]}",
                        "retryable": True,
                    })
                return TransformResult.success(row)

        source = ListSource([{"index": 0}, {"index": 1}])
        transform = FlakyTransform()
        sink = CollectSink()

        retry_config = RetryConfig(
            max_attempts=5,
            initial_delay_ms=1,  # Fast for testing
            max_delay_ms=10,
            exponential_base=2,
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
            retry_config=retry_config,
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config))

        assert result.status == "completed"
        assert result.rows_processed == 2
        assert result.rows_succeeded == 2

        # Both rows should have been retried
        assert attempt_counts[0] == 3
        assert attempt_counts[1] == 3

        # Sink received both rows
        assert len(sink.results) == 2

        # Verify audit trail records retries
        recorder = LandscapeRecorder(db)
        rows = recorder.get_rows(result.run_id)

        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            for token in tokens:
                states = recorder.get_node_states_for_token(token.token_id)
                # Should have multiple attempts recorded
                transform_states = [s for s in states if "flaky" in s.plugin_name]
                # At least one state should show retry metadata
                assert any(s.attempt_number > 1 for s in transform_states if hasattr(s, 'attempt_number'))

    def test_permanent_failure_quarantines_after_max_retries(self) -> None:
        """Transform that always fails should quarantine after max retries.

        Verifies:
        - RetryManager exhausts retries
        - Row is quarantined, not lost
        - Audit trail shows all attempts
        - Other rows still succeed
        """
        # ... similar setup with max_attempts=3 and always-failing transform ...

        assert result.status == "completed"
        assert result.rows_quarantined == 1
        assert result.rows_succeeded == 1  # Other row succeeded
```

**Step 2-5: Standard TDD flow**

---

## Task 4: Test explain() Query End-to-End

**Files:**
- Test: `tests/engine/test_integration.py`

**Step 1: Write the failing test**

```python
# Add new test class

class TestExplainQuery:
    """Tests for explain() audit trail queries."""

    def test_explain_traces_output_to_source(self) -> None:
        """explain() should trace any output row back to its source.

        For auditability, given an output row, we must be able to show:
        - Which source row it came from
        - Every transform it passed through
        - Any routing decisions made
        - The final sink that received it
        """
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        # ... pipeline with multiple transforms and a gate ...

        result = orchestrator.run(config, graph=graph)

        # Get an output token_id (from sink)
        recorder = LandscapeRecorder(db)

        # Find a token that reached the sink
        artifacts = recorder.get_artifacts(result.run_id)
        assert len(artifacts) >= 1

        # Get tokens associated with this artifact
        # (This depends on how artifacts link to tokens)

        # For any token, explain() should return complete lineage
        rows = recorder.get_rows(result.run_id)
        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            for token in tokens:
                # Get complete lineage
                states = recorder.get_node_states_for_token(token.token_id)

                # Should include source
                source_states = [s for s in states if s.node_type == "source"]
                # Note: source might not have node_state, but row should exist

                # Should include all transforms
                transform_states = [s for s in states if s.node_type == "transform"]
                assert len(transform_states) >= 1

                # If routed, should include gate
                gate_states = [s for s in states if s.node_type == "gate"]
                # Check routing events if gate exists
                for gate_state in gate_states:
                    events = recorder.get_routing_events(gate_state.state_id)
                    # Routed tokens have routing events

                # Should reach sink
                sink_states = [s for s in states if s.node_type == "sink"]
                # Terminal tokens reach sink OR are quarantined/consumed

    def test_explain_for_aggregated_row(self) -> None:
        """explain() for aggregated output should show all input rows.

        When an aggregation produces one output from N inputs,
        explain() should trace back to all N source rows.
        """
        # ... setup with aggregation that combines 3 rows into 1 ...

        result = orchestrator.run(config)

        # The aggregated output row should link to all 3 input rows
        # via batch_members table

        recorder = LandscapeRecorder(db)

        # Find the batch
        with db._engine.connect() as conn:
            from sqlalchemy import text

            batches = conn.execute(
                text("SELECT batch_id FROM batches WHERE run_id = :run_id"),
                {"run_id": result.run_id},
            ).fetchall()

            assert len(batches) >= 1

            # Get batch members
            for batch_id, in batches:
                members = recorder.get_batch_members(batch_id)
                # Should have 3 members (the 3 input rows)
                assert len(members) == 3

    def test_explain_for_coalesced_row(self) -> None:
        """explain() for coalesced output should show all branch inputs.

        When a coalesce merges N branch outputs into 1,
        explain() should trace back through all N branches to the fork point.
        """
        # ... setup with fork → coalesce ...

        # The coalesced token should link back to:
        # - The parent token that forked
        # - Both child tokens from the branches

        recorder = LandscapeRecorder(db)

        # Find coalesced tokens
        rows = recorder.get_rows(result.run_id)
        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            for token in tokens:
                if token.parent_token_id is not None:
                    # This is a child or coalesced token
                    # Trace back to parent
                    parent = recorder.get_token(token.parent_token_id)
                    assert parent is not None
```

**Step 2-5: Standard TDD flow**

---

## Task 5: Test Error Recovery Scenarios

**Files:**
- Test: `tests/engine/test_integration.py`

**Step 1: Write the failing test**

```python
# Add new test class

class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_partial_success_continues_processing(self) -> None:
        """Some rows fail, others succeed - pipeline completes.

        Pipeline should:
        - Process all rows
        - Quarantine failures
        - Complete successfully with partial results
        """
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class SelectiveFailTransform(BaseTransform):
            """Fails on even values, succeeds on odd."""
            name = "selective_fail"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                if row["value"] % 2 == 0:
                    return TransformResult.error({"message": "Even values fail"})
                return TransformResult.success(row)

        source = ListSource([{"value": i} for i in range(10)])  # 5 even, 5 odd
        transform = SelectiveFailTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=_build_test_graph(config))

        # Pipeline completes despite failures
        assert result.status == "completed"
        assert result.rows_processed == 10
        assert result.rows_succeeded == 5  # Odd values
        assert result.rows_quarantined == 5  # Even values

        # Sink received successful rows
        assert len(sink.results) == 5
        assert all(r["value"] % 2 == 1 for r in sink.results)

    def test_quarantined_rows_have_audit_trail(self) -> None:
        """Quarantined rows must have complete audit trail.

        Even failed rows must be traceable - we need to know:
        - What source row failed
        - At which transform
        - What the error was
        """
        # ... similar setup ...

        recorder = LandscapeRecorder(db)

        # Find quarantined tokens
        rows = recorder.get_rows(result.run_id)
        quarantined_count = 0

        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            for token in tokens:
                states = recorder.get_node_states_for_token(token.token_id)

                # Check for error states
                error_states = [s for s in states if s.status == "error"]
                if error_states:
                    quarantined_count += 1

                    # Error state should have error details
                    for error_state in error_states:
                        assert error_state.error_message is not None
                        # Should identify which transform failed
                        assert error_state.node_id is not None

        assert quarantined_count == 5  # All even-value rows
```

**Step 2-5: Standard TDD flow**

---

## Task 6: Test RunResult Metrics

**Files:**
- Test: `tests/engine/test_integration.py`

**Step 1: Write the failing test**

```python
# Add to TestComplexDAGIntegration

def test_run_result_captures_all_metrics(self) -> None:
    """RunResult should capture metrics for all operations.

    Verifies RunResult includes:
    - rows_processed
    - rows_succeeded
    - rows_failed / rows_quarantined
    - rows_routed
    - rows_forked
    - rows_coalesced
    - rows_aggregated (batched)
    """
    # ... run complex pipeline with all features ...

    result = orchestrator.run(config, graph=graph)

    # All metrics should be populated
    assert result.rows_processed >= 0
    assert result.rows_succeeded >= 0
    assert result.rows_quarantined >= 0
    assert result.rows_routed >= 0
    assert result.rows_forked >= 0
    assert result.rows_coalesced >= 0

    # Metrics should be consistent
    # rows_processed = rows_succeeded + rows_quarantined + rows_routed + rows_forked
    # (This formula depends on how we count - forked parents vs children, etc.)

    # At minimum, succeeded + quarantined <= processed
    assert result.rows_succeeded + result.rows_quarantined <= result.rows_processed
```

**Step 2-5: Standard TDD flow**

---

## Task 7: Verify All Tests Pass

**Step 1: Run all integration tests**

```bash
pytest tests/engine/test_integration.py -v
pytest tests/plugins/test_integration.py -v
```

**Step 2: Run with coverage**

```bash
pytest tests/engine/test_integration.py tests/plugins/test_integration.py \
    --cov=src/elspeth/engine \
    --cov-report=term-missing
```

Target: >85% coverage for engine module

**Step 3: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test(wp-14d): complete integration test coverage

WP-14d implementation complete:
- Diamond DAG (fork → transform → coalesce)
- Combined features pipeline (gate + fork + agg + coalesce)
- Retry integration end-to-end
- explain() query verification
- Error recovery and quarantine handling
- RunResult metrics validation

All WP-14 test coverage now complete.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Task | Description | Estimated Time |
|------|-------------|----------------|
| 1 | Diamond DAG fork/coalesce | 1 hour |
| 2 | Combined features pipeline | 1.5 hours |
| 3 | Retry integration | 45 min |
| 4 | explain() query verification | 1 hour |
| 5 | Error recovery scenarios | 45 min |
| 6 | RunResult metrics | 30 min |
| 7 | Final verification | 15 min |

**Total estimated time: ~6 hours**

---

## WP-14 Complete Summary

| Sub-package | Focus | Est. Time |
|-------------|-------|-----------|
| **WP-14a** | Fork/Coalesce | 5 hours |
| **WP-14b** | Gates | 3 hours |
| **WP-14c** | Aggregation | 5 hours |
| **WP-14d** | Integration | 6 hours |
| **Total** | | **19 hours** |

> Note: Original estimate was 16 hours. Revised to 19 hours based on detailed analysis. The additional time accounts for complex graph construction helpers needed for diamond DAG and combined feature tests.

---

## Test Coverage Checklist

- [ ] Diamond DAG pattern (fork → transform → coalesce)
- [ ] Combined features pipeline (gate + fork + agg + coalesce)
- [ ] Retry with transient failures
- [ ] Retry exhaustion → quarantine
- [ ] explain() traces output to source
- [ ] explain() for aggregated rows
- [ ] explain() for coalesced rows
- [ ] Partial success (some rows fail, others succeed)
- [ ] Quarantined rows have audit trail
- [ ] RunResult captures all metrics
- [ ] >85% engine module coverage
