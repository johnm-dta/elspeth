# Fork-to-Sink Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the fork implementation so that fork children route to their branch-named sinks instead of all going to the default output sink.

**Problem:** Fork children have `branch_name` set (e.g., "path_a", "path_b") but the orchestrator ignores this and sends ALL `COMPLETED` tokens to `output_sink_name`. This makes fork useless for divergent workflows.

**Root Cause:** `orchestrator.py` doesn't use `branch_name` for sink resolution (lines 595-597)

**Review Status:** Reviewed 2026-01-18, CONDITIONAL GO with changes incorporated below.

---

## The Fix

### Primary Fix: Orchestrator Sink Resolution

**Current (broken):**
```python
# orchestrator.py:595-597
if result.outcome == RowOutcome.COMPLETED:
    rows_succeeded += 1
    pending_tokens[output_sink_name].append(result.token)
```

**Fixed:**
```python
if result.outcome == RowOutcome.COMPLETED:
    rows_succeeded += 1
    # Fork children route to branch-named sink if it exists
    sink_name = output_sink_name
    if (
        result.token.branch_name is not None
        and result.token.branch_name in config.sinks
    ):
        sink_name = result.token.branch_name
    pending_tokens[sink_name].append(result.token)
```

### Optional Enhancement: Graph Edge Creation (for visualization)

**Note:** The orchestrator fix works WITHOUT this change. Edge creation is optional for graph visualization purposes only. The `_route_resolution_map` must NOT be modified - existing "fork" values are correct.

```python
# dag.py - OPTIONAL, only if graph visualization needs fork edges
if target == "fork":
    # Create edges for fork_to destinations that are sinks (visualization only)
    # Do NOT modify _route_resolution_map - keep existing "fork" value
    if gate_config.fork_to:
        for branch in gate_config.fork_to:
            if branch in sink_ids:
                graph.add_edge(
                    gid, sink_ids[branch], label=branch, mode=RoutingMode.COPY
                )
    continue
```

---

## Task 0: Write Failing Tests First (TDD)

**Files:**
- Test: `tests/engine/test_engine_gates.py`

**Step 1: Add failing tests to `TestForkCreatesChildTokens`**

```python
def test_fork_children_route_to_branch_named_sinks(self) -> None:
    """Fork children with branch_name route to matching sinks.

    This is the core fork use case:
    - Gate forks to ["path_a", "path_b"]
    - Child with branch_name="path_a" goes to sink named "path_a"
    - Child with branch_name="path_b" goes to sink named "path_b"
    """
    from elspeth.core.config import (
        DatasourceSettings,
        ElspethSettings,
        GateSettings,
        SinkSettings,
    )
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.artifacts import ArtifactDescriptor
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

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

    class CollectSink(_TestSinkBase):
        name = "collect"
        config: ClassVar[dict[str, Any]] = {}

        def __init__(self) -> None:
            self.results: list[dict[str, Any]] = []

        def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
            self.results.extend(rows)
            return ArtifactDescriptor.for_file(
                path="memory", size_bytes=0, content_hash=""
            )

        def close(self) -> None:
            pass

    source = ListSource([{"value": 42}])
    path_a_sink = CollectSink()
    path_b_sink = CollectSink()

    # Config with fork gate and branch-named sinks
    settings = ElspethSettings(
        datasource=DatasourceSettings(plugin="list_source"),
        sinks={
            "path_a": SinkSettings(plugin="collect"),
            "path_b": SinkSettings(plugin="collect"),
        },
        gates=[
            GateSettings(
                name="forking_gate",
                condition="True",
                routes={"true": "fork"},
                fork_to=["path_a", "path_b"],
            ),
        ],
        output_sink="path_a",  # Default, but fork should override for path_b
    )

    graph = ExecutionGraph.from_config(settings)

    config = PipelineConfig(
        source=source,
        transforms=[],
        sinks={"path_a": path_a_sink, "path_b": path_b_sink},
        gates=settings.gates,
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config, graph=graph)

    assert result.status == "completed"

    # CRITICAL: Each sink gets exactly one row (the fork child for that branch)
    assert len(path_a_sink.results) == 1, f"path_a should get 1 row, got {len(path_a_sink.results)}"
    assert len(path_b_sink.results) == 1, f"path_b should get 1 row, got {len(path_b_sink.results)}"

    # Both should have the same value (forked from same parent)
    assert path_a_sink.results[0]["value"] == 42
    assert path_b_sink.results[0]["value"] == 42


def test_fork_unmatched_branch_falls_back_to_output_sink(self) -> None:
    """Fork child with branch_name not matching any sink goes to output_sink.

    Edge case: fork_to=["stats", "alerts"] but only "alerts" is a sink.
    Child with branch_name="stats" should fall back to output_sink.
    """
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.artifacts import ArtifactDescriptor
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    db = LandscapeDB.in_memory()

    # ... same test infrastructure as above ...

    source = ListSource([{"value": 99}])
    default_sink = CollectSink()  # output_sink
    alerts_sink = CollectSink()   # only one fork branch has matching sink

    # fork_to has "stats" and "alerts", but only "alerts" is a sink
    # "stats" child should fall back to default output_sink
    config = PipelineConfig(
        source=source,
        transforms=[],
        sinks={"default": default_sink, "alerts": alerts_sink},
        gates=[
            GateSettings(
                name="forking_gate",
                condition="True",
                routes={"true": "fork"},
                fork_to=["stats", "alerts"],  # "stats" is NOT a sink
            ),
        ],
    )

    # Build graph manually since "stats" isn't a sink
    graph = _build_fork_graph_with_partial_sinks(config)

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config, graph=graph, output_sink_name="default")

    assert result.status == "completed"

    # "alerts" child → alerts_sink (branch matches sink)
    assert len(alerts_sink.results) == 1

    # "stats" child → default_sink (no matching sink, falls back)
    assert len(default_sink.results) == 1
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/engine/test_engine_gates.py::TestForkCreatesChildTokens::test_fork_children_route_to_branch_named_sinks -v
pytest tests/engine/test_engine_gates.py::TestForkCreatesChildTokens::test_fork_unmatched_branch_falls_back_to_output_sink -v
```

Expected: FAIL - all rows go to output_sink

**Step 3: Commit failing tests**

```bash
git add tests/engine/test_engine_gates.py
git commit -m "$(cat <<'EOF'
test(gates): add failing tests for fork-to-sink routing

Two test cases:
1. Fork children route to branch-named sinks
2. Unmatched branch_name falls back to output_sink

Both will fail until orchestrator fix is implemented.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Fix Orchestrator Sink Resolution

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py` (lines 595-597)

**Step 1: Update COMPLETED token routing**

Replace lines 595-597:

```python
# BEFORE:
if result.outcome == RowOutcome.COMPLETED:
    rows_succeeded += 1
    pending_tokens[output_sink_name].append(result.token)

# AFTER:
if result.outcome == RowOutcome.COMPLETED:
    rows_succeeded += 1
    # Fork children route to branch-named sink if it exists
    sink_name = output_sink_name
    if (
        result.token.branch_name is not None
        and result.token.branch_name in config.sinks
    ):
        sink_name = result.token.branch_name
    pending_tokens[sink_name].append(result.token)
```

**Step 2: Run the failing test - should now pass**

```bash
pytest tests/engine/test_engine_gates.py::TestForkCreatesChildTokens::test_fork_children_route_to_branch_named_sinks -v
```

Expected: PASS

**Step 3: Run all engine tests**

```bash
pytest tests/engine/ -v
```

**Step 4: Commit**

```bash
git add src/elspeth/engine/orchestrator.py
git commit -m "$(cat <<'EOF'
fix(orchestrator): route fork children to branch-named sinks

COMPLETED tokens with branch_name now route to matching sink if one
exists, instead of all going to output_sink.

This completes the fork implementation:
- Gate creates children with branch_name="path_a", "path_b"
- Child with branch_name="path_a" → sink named "path_a"
- Child with branch_name="path_b" → sink named "path_b"

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add RunResult.rows_forked Metric

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py`

**Step 1: Add field to RunResult**

```python
@dataclass
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: RunStatus
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed: int
    rows_quarantined: int = 0
    rows_forked: int = 0  # NEW: Count of parent tokens that forked
```

**Step 2: Increment counter in run loop**

```python
elif result.outcome == RowOutcome.FORKED:
    rows_forked += 1  # NEW: Count forked parents
    # Children are counted separately when they reach terminal state
```

**Step 3: Initialize counter and include in return**

```python
rows_forked = 0  # Add near other counter initializations

# In return statement:
return RunResult(
    run_id=run_id,
    status=...,
    rows_processed=rows_processed,
    rows_succeeded=rows_succeeded,
    rows_failed=rows_failed,
    rows_routed=rows_routed,
    rows_quarantined=rows_quarantined,
    rows_forked=rows_forked,  # NEW
)
```

**Step 4: Run tests and commit**

```bash
pytest tests/engine/ -v
git add src/elspeth/engine/orchestrator.py
git commit -m "$(cat <<'EOF'
feat(orchestrator): add rows_forked metric to RunResult

Track count of parent tokens that forked. Children are counted
separately when they reach their terminal state.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update Test to Use rows_forked

**Files:**
- Modify: `tests/engine/test_engine_gates.py`

**Step 1: Add assertion for rows_forked**

```python
# Add to test_fork_children_route_to_branch_named_sinks
assert result.rows_forked == 1  # One parent forked into two children
```

**Step 2: Run test**

```bash
pytest tests/engine/test_engine_gates.py::TestForkCreatesChildTokens::test_fork_children_route_to_branch_named_sinks -v
```

**Step 3: Commit**

```bash
git add tests/engine/test_engine_gates.py
git commit -m "$(cat <<'EOF'
test(gates): verify rows_forked metric in fork test

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Full Verification

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

**Step 2: Run mypy**

```bash
mypy src/elspeth/engine/orchestrator.py --strict
```

**Step 3: Run ruff**

```bash
ruff check src/elspeth/engine/orchestrator.py
```

---

## Verification Checklist

- [ ] Failing tests written first (TDD)
- [ ] `orchestrator.py` routes COMPLETED tokens by branch_name
- [ ] Unmatched branch_name falls back to output_sink
- [ ] `RunResult.rows_forked` field added
- [ ] Tests pass: fork children go to branch-named sinks
- [ ] All existing tests pass (no regressions)
- [ ] mypy passes
- [ ] ruff passes

---

## Impact Analysis

| Change | Risk | Mitigation |
|--------|------|------------|
| Sink resolution logic | Medium | Check for branch_name AND sink existence; falls back to output_sink |
| RunResult field | None | Additive, defaults to 0 |

**Behavioral change (correct):** Fork children will now go to different sinks instead of all to output_sink. This is the intended behavior - existing code that relied on the broken behavior would need adjustment, but such code would have been working around a bug.
