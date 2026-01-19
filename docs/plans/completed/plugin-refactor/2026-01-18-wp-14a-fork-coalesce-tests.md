# WP-14a: Fork/Coalesce Test Rewrites Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

> **Updated 2026-01-19:** Migrated from `BaseGate` plugin classes to config-driven `GateSettings` for consistency with WP-02 (plugin gate removal) and WP-16 (test cleanup).

**Goal:** Complete test coverage for fork work queue (WP-07) and coalesce executor (WP-08) integration with RowProcessor and Orchestrator.

**Architecture:** Tests will verify the complete token lifecycle through fork → parallel processing → coalesce, ensuring audit trail integrity and correct terminal state propagation. The existing `CoalesceExecutor` unit tests validate executor behavior in isolation; this plan adds integration tests showing how `RowProcessor` orchestrates fork/coalesce operations.

**Tech Stack:** pytest, in-memory LandscapeDB, mock transforms/gates

---

## Current State Analysis

### Existing Coverage ✅
- `test_processor.py`: `TestRowProcessorNestedForks`, `TestRowProcessorWorkQueue` (fork work queue)
- `test_coalesce_executor.py`: All 4 policies, 3 merge strategies, timeout handling
- `test_integration.py`: `test_full_pipeline_with_fork_writes_all_children_to_sink`

### Gaps to Fill 🔴
1. **RowProcessor + CoalesceExecutor integration** - Coalesce unit tests exist but aren't wired through processor
2. **COALESCED terminal outcome** - Not tested through RowProcessor path
3. **Error handling in forked paths** - What happens when one fork child quarantines?
4. **Coalesce with mixed outcomes** - require_all vs best_effort when children have different outcomes
5. **Audit trail lineage for coalesced tokens** - Verify explain() works after coalesce

---

## Task 1: Add CoalesceExecutor to RowProcessor

**Files:**
- Modify: `src/elspeth/engine/processor.py:74-118` (constructor)
- Modify: `src/elspeth/engine/processor.py:238-445` (_process_single_token)
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

```python
# Add to tests/engine/test_processor.py

class TestRowProcessorCoalesce:
    """Test RowProcessor integration with CoalesceExecutor."""

    def test_processor_accepts_coalesce_executor(self) -> None:
        """RowProcessor should accept coalesce_executor parameter."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )

        # Should not raise
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
            coalesce_executor=coalesce_executor,
        )
        assert processor._coalesce_executor is coalesce_executor
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_processor.py::TestRowProcessorCoalesce::test_processor_accepts_coalesce_executor -v`
Expected: FAIL with "TypeError: __init__() got an unexpected keyword argument 'coalesce_executor'"

**Step 3: Write minimal implementation**

Edit `src/elspeth/engine/processor.py` constructor to accept `coalesce_executor`:

```python
# In RowProcessor.__init__ signature (line ~74)
def __init__(
    self,
    recorder: LandscapeRecorder,
    span_factory: SpanFactory,
    run_id: str,
    source_node_id: str,
    *,
    edge_map: dict[tuple[str, str], str] | None = None,
    route_resolution_map: dict[tuple[str, str], str] | None = None,
    config_gates: list[GateSettings] | None = None,
    config_gate_id_map: dict[str, str] | None = None,
    aggregation_settings: dict[str, AggregationSettings] | None = None,
    retry_manager: RetryManager | None = None,
    coalesce_executor: "CoalesceExecutor | None" = None,  # ADD THIS
) -> None:
    # ... existing code ...
    self._coalesce_executor = coalesce_executor  # ADD THIS
```

Add import at top of file:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.engine.coalesce_executor import CoalesceExecutor
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_processor.py::TestRowProcessorCoalesce::test_processor_accepts_coalesce_executor -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor.py
git commit -m "$(cat <<'EOF'
feat(processor): accept coalesce_executor parameter

Add optional coalesce_executor parameter to RowProcessor for fork/coalesce
integration. This is preparation for WP-14a test coverage.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Test Fork → Coalesce Flow (require_all)

**Files:**
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

```python
# Add to TestRowProcessorCoalesce in tests/engine/test_processor.py

def test_fork_then_coalesce_require_all(self) -> None:
    """Fork children should coalesce when all branches arrive."""
    from elspeth.contracts import NodeType, RoutingMode
    from elspeth.core.config import CoalesceSettings
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
    from elspeth.engine.coalesce_executor import CoalesceExecutor
    from elspeth.engine.processor import RowProcessor
    from elspeth.engine.spans import SpanFactory
    from elspeth.engine.tokens import TokenManager
    from elspeth.plugins.results import GateResult, RoutingAction, RowOutcome

    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    run = recorder.begin_run(config={}, canonical_version="v1")
    token_manager = TokenManager(recorder)

    # Register nodes
    source = recorder.register_node(
        run_id=run.run_id,
        plugin_name="source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
    fork_gate = recorder.register_node(
        run_id=run.run_id,
        plugin_name="splitter",
        node_type=NodeType.GATE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
    transform_a = recorder.register_node(
        run_id=run.run_id,
        plugin_name="enrich_a",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
    transform_b = recorder.register_node(
        run_id=run.run_id,
        plugin_name="enrich_b",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
    coalesce_node = recorder.register_node(
        run_id=run.run_id,
        plugin_name="merger",
        node_type=NodeType.COALESCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )

    # Register edges for fork paths
    edge_a = recorder.register_edge(
        run_id=run.run_id,
        from_node_id=fork_gate.node_id,
        to_node_id=transform_a.node_id,
        label="path_a",
        mode=RoutingMode.COPY,
    )
    edge_b = recorder.register_edge(
        run_id=run.run_id,
        from_node_id=fork_gate.node_id,
        to_node_id=transform_b.node_id,
        label="path_b",
        mode=RoutingMode.COPY,
    )

    # Setup coalesce executor
    coalesce_settings = CoalesceSettings(
        name="merger",
        branches=["path_a", "path_b"],
        policy="require_all",
        merge="union",
    )
    coalesce_executor = CoalesceExecutor(
        recorder=recorder,
        span_factory=SpanFactory(),
        token_manager=token_manager,
        run_id=run.run_id,
    )
    coalesce_executor.register_coalesce(coalesce_settings, coalesce_node.node_id)

    # Create config-driven fork gate (not a plugin class)
    fork_gate_settings = GateSettings(
        name="splitter",
        condition="True",  # Always fork
        routes={"true": "fork"},
        fork_to=["path_a", "path_b"],
    )

    class EnrichA(BaseTransform):
        name = "enrich_a"
        input_schema = _TestSchema
        output_schema = _TestSchema

        def __init__(self, node_id: str) -> None:
            super().__init__({})
            self.node_id = node_id

        def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
            return TransformResult.success({**row, "sentiment": "positive"})

    class EnrichB(BaseTransform):
        name = "enrich_b"
        input_schema = _TestSchema
        output_schema = _TestSchema

        def __init__(self, node_id: str) -> None:
            super().__init__({})
            self.node_id = node_id

        def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
            return TransformResult.success({**row, "entities": ["ACME"]})

    processor = RowProcessor(
        recorder=recorder,
        span_factory=SpanFactory(),
        run_id=run.run_id,
        source_node_id=source.node_id,
        edge_map={
            (fork_gate.node_id, "path_a"): edge_a.edge_id,
            (fork_gate.node_id, "path_b"): edge_b.edge_id,
        },
        config_gates=[fork_gate_settings],  # Config-driven gate
        config_gate_id_map={"splitter": fork_gate.node_id},
        route_resolution_map={
            (fork_gate.node_id, "path_a"): "fork",
            (fork_gate.node_id, "path_b"): "fork",
        },
        coalesce_executor=coalesce_executor,
        coalesce_node_ids={"merger": coalesce_node.node_id},
    )

    ctx = PluginContext(run_id=run.run_id, config={})

    # Process should:
    # 1. Fork at config gate (parent FORKED)
    # 2. Process path_a (add sentiment)
    # 3. Process path_b (add entities)
    # 4. Coalesce both paths (merged token COALESCED)
    results = processor.process_row(
        row_index=0,
        row_data={"text": "ACME earnings"},
        transforms=[
            EnrichA(transform_a.node_id),
            EnrichB(transform_b.node_id),
        ],
        ctx=ctx,
        coalesce_at_step=2,  # After both transforms (gate runs before transforms)
        coalesce_name="merger",
    )

    # Verify outcomes
    outcomes = {r.outcome for r in results}
    assert RowOutcome.FORKED in outcomes
    assert RowOutcome.COALESCED in outcomes

    # Find the coalesced result
    coalesced = [r for r in results if r.outcome == RowOutcome.COALESCED]
    assert len(coalesced) == 1

    # Verify merged data
    merged_data = coalesced[0].final_data
    assert merged_data["sentiment"] == "positive"
    assert merged_data["entities"] == ["ACME"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_processor.py::TestRowProcessorCoalesce::test_fork_then_coalesce_require_all -v`
Expected: FAIL - RowProcessor doesn't yet support coalesce integration

**Step 3: Write implementation (larger change - see Task 3)**

This test will drive the implementation in Task 3.

**Step 4: Run test to verify it passes**

Run after Task 3 implementation.

**Step 5: Commit**

```bash
git add tests/engine/test_processor.py
git commit -m "$(cat <<'EOF'
test(processor): add fork → coalesce require_all test

Tests the complete flow: fork at gate → parallel transforms → coalesce.
Verifies FORKED and COALESCED outcomes, merged data contains fields from
both paths.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement Coalesce Integration in RowProcessor

**Files:**
- Modify: `src/elspeth/engine/processor.py`

**Step 1: The test from Task 2 is already written**

**Step 2: Already verified it fails**

**Step 3: Implement coalesce handling in _process_single_token**

The implementation needs to:
1. Accept `coalesce_at_step` and `coalesce_name` parameters
2. After processing a fork child, check if it should be coalesced
3. Call coalesce_executor.accept() with the processed token
4. If held, return early (waiting for siblings)
5. If merged, continue processing with merged token

```python
# In RowProcessor._process_single_token, after transforms complete but before
# returning COMPLETED, add coalesce handling:

# Check if this token should be coalesced
if (
    self._coalesce_executor is not None
    and current_token.branch_name is not None  # Is a fork child
    and coalesce_name is not None
    and coalesce_at_step is not None
    and step >= coalesce_at_step
):
    outcome = self._coalesce_executor.accept(
        token=current_token,
        coalesce_name=coalesce_name,
        step_in_pipeline=step,
    )

    if outcome.held:
        # Waiting for siblings - this token is consumed
        return (
            RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=RowOutcome.CONSUMED_IN_COALESCE,  # New intermediate state
            ),
            child_items,
        )
    else:
        # Merged - return COALESCED with merged data
        assert outcome.merged_token is not None
        return (
            RowResult(
                token=outcome.merged_token,
                final_data=outcome.merged_token.row_data,
                outcome=RowOutcome.COALESCED,
            ),
            child_items,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_processor.py::TestRowProcessorCoalesce -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/processor.py
git commit -m "$(cat <<'EOF'
feat(processor): integrate CoalesceExecutor for fork/join

Add coalesce handling to RowProcessor._process_single_token:
- Accept coalesce parameters (coalesce_at_step, coalesce_name)
- Fork children submit to CoalesceExecutor after processing
- Held tokens return early, merged tokens return COALESCED outcome

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Test Coalesce with Mixed Outcomes (best_effort)

**Files:**
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

```python
def test_coalesce_best_effort_with_quarantined_child(self) -> None:
    """best_effort policy should merge available children even if one quarantines.

    Scenario:
    - Fork to 3 paths: sentiment, entities, summary
    - summary path quarantines (simulated error)
    - best_effort timeout triggers, merges sentiment + entities
    - Result should be COALESCED with partial data
    """
    # ... similar setup to test_fork_then_coalesce_require_all ...

    # Create transform that returns error (will quarantine)
    class FailingSummary(BaseTransform):
        name = "summary"
        input_schema = _TestSchema
        output_schema = _TestSchema
        _on_error = "discard"  # Quarantine on error

        def __init__(self, node_id: str) -> None:
            super().__init__({})
            self.node_id = node_id

        def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
            return TransformResult.error({"message": "Summary failed"})

    # Setup coalesce with best_effort policy and short timeout
    coalesce_settings = CoalesceSettings(
        name="merger",
        branches=["sentiment", "entities", "summary"],
        policy="best_effort",
        timeout_seconds=0.1,
        merge="union",
    )
    # ... rest of setup ...

    results = processor.process_row(...)

    # Verify:
    # - 1 FORKED (parent)
    # - 1 QUARANTINED (summary path)
    # - 1 COALESCED (merged sentiment + entities)
    outcomes = [r.outcome for r in results]
    assert outcomes.count(RowOutcome.FORKED) == 1
    assert outcomes.count(RowOutcome.QUARANTINED) == 1
    assert outcomes.count(RowOutcome.COALESCED) == 1

    # Merged data has sentiment and entities, not summary
    coalesced = [r for r in results if r.outcome == RowOutcome.COALESCED][0]
    assert "sentiment" in coalesced.final_data
    assert "entities" in coalesced.final_data
    assert "summary" not in coalesced.final_data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_processor.py::TestRowProcessorCoalesce::test_coalesce_best_effort_with_quarantined_child -v`
Expected: FAIL - best_effort timeout handling not yet implemented in processor flow

**Step 3: Implementation**

This requires adding timeout handling to the processor - checking for timed-out coalesces after each work queue iteration.

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add tests/engine/test_processor.py src/elspeth/engine/processor.py
git commit -m "$(cat <<'EOF'
feat(processor): support best_effort coalesce with quarantined children

Add timeout handling for best_effort coalesce policy. When children are
quarantined or fail to arrive, best_effort merges whatever is available.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Test Coalesce Audit Trail (explain query)

**Files:**
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

```python
def test_coalesced_token_audit_trail_complete(self) -> None:
    """Coalesced tokens should have complete audit trail for explain().

    After fork → process → coalesce, querying explain() on the merged
    token should show:
    - Original source row
    - Fork point
    - Both branch processing steps
    - Coalesce point with merge metadata
    """
    # ... setup similar to test_fork_then_coalesce_require_all ...

    results = processor.process_row(...)

    # Get the coalesced token
    coalesced = [r for r in results if r.outcome == RowOutcome.COALESCED][0]

    # Query audit trail
    node_states = recorder.get_node_states_for_token(coalesced.token_id)

    # Should have states for:
    # - Fork gate
    # - Transform A (path_a)
    # - Transform B (path_b)
    # - Coalesce node
    node_ids = {s.node_id for s in node_states}
    assert fork_gate.node_id in node_ids
    assert coalesce_node.node_id in node_ids

    # Verify coalesce metadata was recorded
    coalesce_states = [s for s in node_states if s.node_id == coalesce_node.node_id]
    assert len(coalesce_states) == 1

    # Should have coalesce-specific metadata
    coalesce_state = coalesce_states[0]
    assert coalesce_state.status.value == "success"
    # Metadata should include policy, merge_strategy, branches_arrived
```

**Step 2-5: Standard TDD flow**

---

## Task 6: Test Quorum Policy (2 of 3)

**Files:**
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

```python
def test_coalesce_quorum_merges_at_threshold(self) -> None:
    """Quorum policy should merge when quorum_count branches arrive.

    Setup: Fork to 3 paths, quorum=2
    - When 2 of 3 arrive, merge immediately
    - 3rd branch result is discarded (arrives after merge)
    """
    coalesce_settings = CoalesceSettings(
        name="merger",
        branches=["fast", "medium", "slow"],
        policy="quorum",
        quorum_count=2,
        merge="nested",
    )

    # ... setup ...

    # Process order: fast completes, medium completes (triggers merge)
    # slow completes but too late

    results = processor.process_row(...)

    # Verify COALESCED contains data from first 2 arrivals
    coalesced = [r for r in results if r.outcome == RowOutcome.COALESCED][0]
    assert "fast" in coalesced.final_data  # nested structure
    assert "medium" in coalesced.final_data
    # slow may or may not be present depending on timing
```

**Step 2-5: Standard TDD flow**

---

## Task 7: Test Nested Fork → Coalesce (Complex DAG)

**Files:**
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

```python
def test_nested_fork_coalesce(self) -> None:
    """Test fork within fork, with coalesce at each level.

    DAG structure:
    source → gate1 (fork A,B) → [
        path_a → gate2 (fork A1,A2) → [A1, A2] → coalesce_inner → ...
        path_b → transform_b
    ] → coalesce_outer

    Should produce:
    - 1 parent FORKED (gate1)
    - 2 level-1 children (path_a FORKED, path_b continues)
    - 2 level-2 children from path_a (A1, A2)
    - 1 inner COALESCED (A1+A2)
    - 1 outer COALESCED (inner+path_b)
    """
    # This is the complex case that validates full DAG support
    # ... implementation ...
```

**Step 2-5: Standard TDD flow**

---

## Task 8: Integration Test - Full Pipeline with Sink

**Files:**
- Test: `tests/engine/test_integration.py`

**Step 1: Write the failing test**

```python
class TestForkCoalescePipelineIntegration:
    """End-to-end fork → coalesce → sink tests."""

    def test_fork_coalesce_writes_merged_to_sink(self) -> None:
        """Complete pipeline: source → fork → process → coalesce → sink.

        Verifies:
        - Sink receives merged data
        - Only 1 row written to sink (not 2 fork children separately)
        - Sink artifact has correct content hash
        """
        # ... full pipeline setup with real CSV sink ...

        orchestrator.run()

        # Verify sink received merged data
        assert len(sink.rows_written) == 1
        row = sink.rows_written[0]
        assert row["sentiment"] == "positive"
        assert row["entities"] == ["ACME"]

        # Verify artifact recorded
        artifact = run_result.artifacts["output"]
        assert artifact.content_hash is not None
```

**Step 2-5: Standard TDD flow**

---

## Task 9: Verify Tests Pass and Coverage

**Step 1: Run all fork/coalesce tests**

```bash
pytest tests/engine/test_processor.py::TestRowProcessorCoalesce -v
pytest tests/engine/test_processor.py::TestRowProcessorNestedForks -v
pytest tests/engine/test_processor.py::TestRowProcessorWorkQueue -v
pytest tests/engine/test_coalesce_executor.py -v
pytest tests/engine/test_integration.py::TestForkCoalescePipelineIntegration -v
```

**Step 2: Verify coverage**

```bash
pytest tests/engine/test_processor.py tests/engine/test_coalesce_executor.py --cov=src/elspeth/engine/processor --cov=src/elspeth/engine/coalesce_executor --cov-report=term-missing
```

Target: >90% coverage for fork/coalesce paths in processor.py and coalesce_executor.py

**Step 3: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test(wp-14a): complete fork/coalesce test coverage

WP-14a implementation complete:
- RowProcessor + CoalesceExecutor integration
- require_all, best_effort, quorum, first policies through processor
- Mixed outcomes (quarantine in fork path)
- Nested fork/coalesce DAG
- Audit trail verification for coalesced tokens
- Full pipeline integration with sink

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Task | Description | Estimated Time |
|------|-------------|----------------|
| 1 | Add CoalesceExecutor to RowProcessor constructor | 15 min |
| 2 | Test fork → coalesce (require_all) | 30 min |
| 3 | Implement coalesce integration | 1 hour |
| 4 | Test best_effort with quarantined child | 45 min |
| 5 | Test audit trail for coalesced tokens | 30 min |
| 6 | Test quorum policy | 30 min |
| 7 | Test nested fork/coalesce | 45 min |
| 8 | Integration test with sink | 30 min |
| 9 | Coverage verification | 15 min |

**Total estimated time: ~5 hours**

---

## Test Coverage Checklist

- [ ] RowProcessor accepts coalesce_executor parameter
- [ ] Fork → coalesce with require_all policy
- [ ] Fork → coalesce with best_effort policy (timeout)
- [ ] Fork → coalesce with quorum policy
- [ ] Fork → coalesce with first policy
- [ ] Mixed outcomes: one child quarantines
- [ ] Nested fork/coalesce DAG
- [ ] Audit trail complete for coalesced tokens
- [ ] Coalesce metadata recorded (policy, branches, timing)
- [ ] Full pipeline integration (source → fork → coalesce → sink)
- [ ] >90% coverage for fork/coalesce code paths
