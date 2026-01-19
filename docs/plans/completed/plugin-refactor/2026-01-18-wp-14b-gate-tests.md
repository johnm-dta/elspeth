# WP-14b: Gate Test Rewrites Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete test coverage for engine-level gates (WP-09), focusing on integration gaps not covered by existing unit tests.

**Architecture:** WP-09 implemented config-driven gates with a safe AST-based expression parser. The existing tests (`test_expression_parser.py`, `test_engine_gates.py`, `test_config_gates.py`) provide ~2700 lines of comprehensive coverage. This plan addresses remaining integration gaps: fork execution through config gates, audit trail verification, and error handling.

**Tech Stack:** pytest, Hypothesis (fuzz testing already exists), in-memory LandscapeDB

**PREREQUISITE:** Execute `2026-01-18-fork-sink-routing.md` FIRST. Task 1 (fork execution tests) depends on fork-to-sink routing being implemented.

---

## Current State Analysis

### Existing Coverage ✅ (Excellent!)

| File | Lines | Coverage |
|------|-------|----------|
| `test_expression_parser.py` | ~960 | Parser operations, security, fuzz testing (1000+ inputs) |
| `test_engine_gates.py` | ~1089 | WP-09 verification, composite conditions, route resolution |
| `test_config_gates.py` | ~657 | Integration, ExecutionGraph.from_config(), multi-gate |

**Total: ~2700 lines of gate tests already exist.**

### Gaps to Fill 🔴

1. **Config gate fork execution** - WP-07 complete, now test fork_to actually creates and processes children
2. **Audit trail for gate decisions** - Verify node_state records contain gate evaluation metadata
3. **Runtime condition errors** - KeyError for missing field, type errors during evaluation
4. **Config gate + plugin gate interaction** - Mixed pipeline with both gate types
5. **Gate condition returning non-boolean** - Expression returns string/int route label

---

## Task 1: Test Config Gate Fork Execution

**Files:**
- Test: `tests/engine/test_engine_gates.py`

**Context:** `TestForkCreatesChildTokens` currently only tests configuration, noting "Fork execution is deferred to WP-07". Now that WP-07 is complete, add execution tests.

**Step 1: Write the failing test**

```python
# Add to TestForkCreatesChildTokens in tests/engine/test_engine_gates.py

def test_config_gate_fork_executes_children(self) -> None:
    """Verify fork_to config gate actually executes child tokens.

    Now that WP-07 (Fork Work Queue) is complete, config gate forks
    should create children that execute through their paths and reach sinks.
    """
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.artifacts import ArtifactDescriptor
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from elspeth.plugins.results import RowOutcome

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

    # Config gate that forks to two paths
    gate = GateSettings(
        name="forking_gate",
        condition="True",  # Always fork
        routes={"all": "fork"},
        fork_to=["path_a", "path_b"],
    )

    config = PipelineConfig(
        source=source,
        transforms=[],
        sinks={"path_a": path_a_sink, "path_b": path_b_sink},
        gates=[gate],
    )

    # Build graph with fork edges
    graph = _build_fork_graph_for_config_gate(config, gate)

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config, graph=graph)

    assert result.status == "completed"
    assert result.rows_processed == 1

    # Both sinks should receive the forked row
    assert len(path_a_sink.results) == 1
    assert len(path_b_sink.results) == 1
    assert path_a_sink.results[0]["value"] == 42
    assert path_b_sink.results[0]["value"] == 42

    # Verify fork was recorded in audit trail
    # Should have: 1 FORKED parent + 2 COMPLETED children
    assert result.rows_forked == 1  # Parent
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_engine_gates.py::TestForkCreatesChildTokens::test_config_gate_fork_executes_children -v`
Expected: FAIL - graph builder function doesn't exist yet

**Step 3: Implement helper function**

```python
def _build_fork_graph_for_config_gate(
    config: PipelineConfig,
    fork_gate: GateSettings,
) -> ExecutionGraph:
    """Build a graph with fork edges for a config gate.

    Creates: source -> config_gate -> [path_a_sink, path_b_sink]
    """
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()

    # Add source
    graph.add_node("source", node_type="source", plugin_name=config.source.name)

    # Add config gate
    gate_id = f"config_gate_{fork_gate.name}"
    graph.add_node(
        gate_id,
        node_type="gate",
        plugin_name=f"config_gate:{fork_gate.name}",
        config={
            "condition": fork_gate.condition,
            "routes": dict(fork_gate.routes),
            "fork_to": fork_gate.fork_to,
        },
    )
    graph.add_edge("source", gate_id, label="continue", mode=RoutingMode.MOVE)

    # Add sinks for each fork path
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)

    # Add fork edges
    for path in fork_gate.fork_to:
        if path in sink_ids:
            graph.add_edge(gate_id, sink_ids[path], label=path, mode=RoutingMode.COPY)

    # Populate internal maps
    graph._sink_id_map = sink_ids
    graph._config_gate_id_map = {fork_gate.name: gate_id}
    graph._route_resolution_map = {
        (gate_id, path): path for path in fork_gate.fork_to
    }

    return graph
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_engine_gates.py::TestForkCreatesChildTokens::test_config_gate_fork_executes_children -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/engine/test_engine_gates.py
git commit -m "$(cat <<'EOF'
test(gates): verify config gate fork executes children

With WP-07 (Fork Work Queue) complete, config gate fork_to now creates
child tokens that execute through their paths. Tests verify both children
reach their designated sinks.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Test Gate Evaluation Audit Trail

**Files:**
- Test: `tests/engine/test_engine_gates.py`

**Step 1: Write the failing test**

```python
# Add to TestEndToEndPipeline in tests/engine/test_engine_gates.py

def test_gate_audit_trail_includes_evaluation_metadata(self) -> None:
    """Gate evaluation should record condition, result, and route in audit trail.

    For auditability, every gate decision must be traceable:
    - What condition was evaluated
    - What the result was (true/false/string)
    - What route was taken
    """
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
    from elspeth.engine.artifacts import ArtifactDescriptor
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    db = LandscapeDB.in_memory()

    class RowSchema(PluginSchema):
        confidence: float

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

    source = ListSource([{"confidence": 0.9}])  # Will pass threshold
    high_sink = CollectSink()
    low_sink = CollectSink()

    gate = GateSettings(
        name="confidence_gate",
        condition="row['confidence'] >= 0.85",
        routes={"true": "continue", "false": "low"},
    )

    config = PipelineConfig(
        source=source,
        transforms=[],
        sinks={"default": high_sink, "low": low_sink},
        gates=[gate],
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(
        config, graph=_build_test_graph_with_config_gates(config)
    )

    # Query node_states for gate evaluation
    recorder = LandscapeRecorder(db)

    # Get all tokens from this run
    with db._engine.connect() as conn:
        from sqlalchemy import text

        # Find the gate node_id
        gate_node = conn.execute(
            text("""
                SELECT node_id FROM nodes
                WHERE run_id = :run_id AND plugin_name = :plugin_name
            """),
            {"run_id": result.run_id, "plugin_name": "config_gate:confidence_gate"},
        ).fetchone()

        assert gate_node is not None, "Gate node should be registered"

        # Get node_states for this gate
        states = conn.execute(
            text("""
                SELECT status, output_hash, metadata_json
                FROM node_states
                WHERE node_id = :node_id
            """),
            {"node_id": gate_node[0]},
        ).fetchall()

        assert len(states) == 1, "Should have one gate evaluation"

        state = states[0]
        assert state[0] == "success", "Gate should succeed"

        # Verify metadata contains gate-specific info
        import json
        metadata = json.loads(state[2]) if state[2] else {}

        # Gate metadata should include:
        assert "condition" in metadata or "route_taken" in metadata
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_engine_gates.py::TestEndToEndPipeline::test_gate_audit_trail_includes_evaluation_metadata -v`
Expected: FAIL or incomplete metadata

**Step 3: Implementation (if needed)**

The `GateExecutor.execute_config_gate` may need to record additional metadata. Check `src/elspeth/engine/executors.py`.

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add tests/engine/test_engine_gates.py src/elspeth/engine/executors.py
git commit -m "$(cat <<'EOF'
test(gates): verify audit trail includes gate evaluation metadata

Gate decisions must be auditable. Tests verify node_states records include
condition evaluated and route taken for explain() queries.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Test Runtime Condition Errors

**Files:**
- Test: `tests/engine/test_engine_gates.py`

**Step 1: Write the failing test**

```python
# Add new test class to tests/engine/test_engine_gates.py

class TestGateRuntimeErrors:
    """Test gate behavior when condition evaluation fails at runtime."""

    def test_missing_field_raises_key_error(self) -> None:
        """Gate condition referencing missing field should fail clearly.

        Per Three-Tier Trust Model: Row data is Tier 2 (elevated trust).
        Missing fields are bugs in upstream transform or source, not
        something to silently handle.
        """
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            other_field: str

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

        # Row has 'other_field' but gate expects 'missing_field'
        source = ListSource([{"other_field": "value"}])
        sink = CollectSink()

        gate = GateSettings(
            name="bad_gate",
            condition="row['missing_field'] > 0",  # Field doesn't exist!
            routes={"true": "continue", "false": "continue"},
        )

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)

        # Should fail with clear error, not silently continue
        with pytest.raises(KeyError) as exc_info:
            orchestrator.run(
                config, graph=_build_test_graph_with_config_gates(config)
            )

        assert "missing_field" in str(exc_info.value)

    def test_optional_field_with_get_succeeds(self) -> None:
        """Gate using row.get() for optional field should succeed.

        This is the correct pattern for optional fields - use row.get()
        with a default, not direct subscript access.
        """
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            other_field: str

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

        source = ListSource([{"other_field": "value"}])
        sink = CollectSink()

        # Use row.get() with default - safe pattern
        gate = GateSettings(
            name="safe_gate",
            condition="row.get('optional_field', 0) > 0",
            routes={"true": "continue", "false": "continue"},
        )

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config, graph=_build_test_graph_with_config_gates(config)
        )

        # Should succeed - row.get() returns 0, 0 > 0 is false, continues
        assert result.status == "completed"
        assert len(sink.results) == 1
```

**Step 2-5: Standard TDD flow**

---

## Task 4: Test Plugin Gate + Config Gate Interaction

**Files:**
- Test: `tests/engine/test_engine_gates.py`

**Step 1: Write the failing test**

```python
# Add new test class to tests/engine/test_engine_gates.py

class TestMixedGatePipeline:
    """Test pipelines with both plugin gates and config gates."""

    def test_plugin_gate_then_config_gate(self) -> None:
        """Pipeline: transform → plugin_gate → config_gate → sink.

        Verifies that plugin gates (BaseGate subclasses) and config gates
        (GateSettings) can coexist in the same pipeline.
        """
        from elspeth.contracts import TransformResult
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.base import BaseGate
        from elspeth.plugins.results import GateResult, RoutingAction

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

        class EnrichTransform(BaseTransform):
            """Adds enriched flag."""
            name = "enrich"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def process(self, row: dict[str, Any], ctx: Any) -> TransformResult:
                return TransformResult.success({**row, "enriched": True})

        class PluginGate(BaseGate):
            """Plugin gate that routes low values."""
            name = "plugin_gate"
            input_schema = RowSchema
            output_schema = RowSchema
            plugin_version = "1.0.0"

            def evaluate(self, row: dict[str, Any], ctx: Any) -> GateResult:
                if row["value"] < 10:
                    return GateResult(row=row, action=RoutingAction.route("too_low"))
                return GateResult(row=row, action=RoutingAction.continue_())

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

        # Rows: 5 (too_low via plugin gate), 50 (high via config gate), 25 (mid via config gate)
        source = ListSource([{"value": 5}, {"value": 50}, {"value": 25}])
        plugin_gate = PluginGate(config={})

        low_sink = CollectSink()
        high_sink = CollectSink()
        mid_sink = CollectSink()

        # Config gate for second-level routing
        config_gate = GateSettings(
            name="level_gate",
            condition="row['value'] > 30",
            routes={"true": "high", "false": "mid"},
        )

        config = PipelineConfig(
            source=source,
            transforms=[EnrichTransform(config={}), plugin_gate],
            sinks={"too_low": low_sink, "high": high_sink, "mid": mid_sink},
            gates=[config_gate],
        )

        # Build graph with plugin gate edge
        graph = _build_mixed_gate_graph(config, plugin_gate, config_gate)

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph)

        assert result.status == "completed"
        assert result.rows_processed == 3

        # 5 → too_low (plugin gate)
        assert len(low_sink.results) == 1
        assert low_sink.results[0]["value"] == 5

        # 50 → high (config gate true)
        assert len(high_sink.results) == 1
        assert high_sink.results[0]["value"] == 50

        # 25 → mid (config gate false)
        assert len(mid_sink.results) == 1
        assert mid_sink.results[0]["value"] == 25

        # All should have enriched flag
        for sink in [low_sink, high_sink, mid_sink]:
            for row in sink.results:
                assert row["enriched"] is True
```

**Step 2-5: Standard TDD flow**

---

## Task 5: Test Non-Boolean Condition Results

**Files:**
- Test: `tests/engine/test_config_gates.py`

**Context:** Existing tests cover string route labels, but should verify edge cases.

**Step 1: Write the failing test**

```python
# Add to TestConfigGateIntegration in tests/engine/test_config_gates.py

def test_config_gate_integer_route_label(self) -> None:
    """Gate condition returning integer should match route label.

    Example: row['priority'] returns 1, 2, or 3 → routes to priority_1, priority_2, priority_3
    """
    from elspeth.core.config import (
        DatasourceSettings,
        ElspethSettings,
        SinkSettings,
    )
    from elspeth.core.config import GateSettings as GateSettingsConfig
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.artifacts import ArtifactDescriptor
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    db = LandscapeDB.in_memory()

    class RowSchema(PluginSchema):
        priority: int

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

    source = ListSource([
        {"priority": 1},
        {"priority": 2},
        {"priority": 1},
    ])
    p1_sink = CollectSink()
    p2_sink = CollectSink()

    # Integer route labels
    settings = ElspethSettings(
        datasource=DatasourceSettings(plugin="csv"),
        sinks={
            "priority_1": SinkSettings(plugin="csv"),
            "priority_2": SinkSettings(plugin="csv"),
        },
        output_sink="priority_1",
        gates=[
            GateSettingsConfig(
                name="priority_router",
                condition="row['priority']",  # Returns 1 or 2
                routes={
                    1: "priority_1",  # Integer key
                    2: "priority_2",
                },
            ),
        ],
    )

    graph = ExecutionGraph.from_config(settings)

    config = PipelineConfig(
        source=source,
        transforms=[],
        sinks={"priority_1": p1_sink, "priority_2": p2_sink},
        gates=settings.gates,
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config, graph=graph)

    assert result.status == "completed"
    assert len(p1_sink.results) == 2  # priority=1 rows
    assert len(p2_sink.results) == 1  # priority=2 row
```

**Step 2-5: Standard TDD flow**

---

## Task 6: Verify Tests Pass and Coverage

**Step 1: Run all gate tests**

```bash
pytest tests/engine/test_expression_parser.py -v
pytest tests/engine/test_engine_gates.py -v
pytest tests/engine/test_config_gates.py -v
```

**Step 2: Verify coverage**

```bash
pytest tests/engine/test_expression_parser.py tests/engine/test_engine_gates.py tests/engine/test_config_gates.py \
    --cov=src/elspeth/engine/expression_parser \
    --cov=src/elspeth/engine/executors \
    --cov-report=term-missing
```

Target: >90% coverage for expression_parser.py, GateExecutor in executors.py

**Step 3: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test(wp-14b): complete gate test coverage

WP-14b implementation complete:
- Config gate fork execution (now that WP-07 is done)
- Audit trail verification for gate decisions
- Runtime condition error handling (KeyError for missing fields)
- Plugin gate + config gate mixed pipeline
- Non-boolean condition results (integer route labels)

Total gate test coverage: ~3000+ lines across 3 test files.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Task | Description | Estimated Time |
|------|-------------|----------------|
| 1 | Config gate fork execution | 45 min |
| 2 | Gate evaluation audit trail | 30 min |
| 3 | Runtime condition errors | 30 min |
| 4 | Plugin gate + config gate interaction | 45 min |
| 5 | Non-boolean condition results | 20 min |
| 6 | Coverage verification | 15 min |

**Total estimated time: ~3 hours**

> **Note:** WP-14b is smaller than expected because WP-09 already delivered comprehensive gate testing (~2700 lines). This plan addresses remaining integration gaps.

---

## Test Coverage Checklist

- [ ] Config gate fork_to actually executes children (WP-07 integration)
- [ ] Gate audit trail includes condition and route taken
- [ ] Missing field in condition raises KeyError (not silently handled)
- [ ] row.get() pattern works for optional fields
- [ ] Plugin gate + config gate in same pipeline
- [ ] Integer route labels work
- [ ] >90% coverage for expression parser and gate executor
