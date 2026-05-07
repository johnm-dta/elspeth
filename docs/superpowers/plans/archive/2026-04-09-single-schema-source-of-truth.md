# Single Schema Source of Truth — Eliminate Dual Schema Representation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `output_schema_config` (typed `SchemaConfig`) the single source of truth for schema data on all nodes, eliminating the parallel `config["schema"]` (untyped dict) representation and the fallback chains that keep them in sync.

**Architecture:** Currently, schema data lives in two places per node — `NodeInfo.output_schema_config` (typed, sometimes None) and `NodeInfo.config["schema"]` (dict, always present). Four separate fallback chains implement "prefer typed, fall back to dict" logic. The fix populates `output_schema_config` at construction time for ALL node types, then collapses the fallback chains to direct reads. `config["schema"]` remains in the config dict (it's part of node ID hashes for sources/transforms/aggregations) but is never read as a schema source.

**Tech Stack:** Python dataclasses, `SchemaConfig.from_dict()`, `object.__setattr__()` (frozen dataclass mutation)

**Key Invariant:** `config["schema"]` stays in the config dict where it was placed by YAML config loading (sources, transforms, sinks, aggregations). We only stop **writing** it on pass-through nodes (gates, coalesce) and stop **reading** it as a schema source everywhere.

**Files Changed:**

| File | Change |
|------|--------|
| `src/elspeth/core/dag/builder.py` | Populate `output_schema_config` for sources, shape-preserving transforms, aggregations; simplify `_assign_schema`; simplify/inline `_best_schema_config`; simplify gate schema guard |
| `src/elspeth/core/dag/graph.py` | Simplify `get_schema_config_from_node()` — direct read, no fallback |
| `src/elspeth/engine/orchestrator/core.py` | Simplify schema_config resolution — direct read, no fallback |
| `tests/unit/core/test_dag_schema_propagation.py` | Update tests to assert `output_schema_config` instead of `config["schema"]` on pass-through nodes; remove fallback tests; remove aliasing tests (frozen SchemaConfig = safe to share) |

---

### Task 1: Populate `output_schema_config` for source nodes

**Files:**
- Modify: `src/elspeth/core/dag/builder.py:154-163`
- Test: `tests/unit/core/test_dag_schema_propagation.py`

- [ ] **Step 1: Write the failing test**

Add to `TestOutputSchemaConfigPropagation` in `tests/unit/core/test_dag_schema_propagation.py`:

```python
def test_source_node_has_output_schema_config(self) -> None:
    """Source nodes should have output_schema_config populated from config['schema']."""
    source = MockSource()

    graph = ExecutionGraph.from_plugin_instances(
        source=source,  # type: ignore[arg-type]
        source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
        transforms=[],
        sinks={"output": MockSink()},  # type: ignore[dict-item]
        aggregations={},
        gates=[],
    )

    source_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.SOURCE]
    assert len(source_nodes) == 1

    node_info = source_nodes[0]
    assert node_info.output_schema_config is not None
    assert node_info.output_schema_config.mode == "observed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_schema_propagation.py::TestOutputSchemaConfigPropagation::test_source_node_has_output_schema_config -v`
Expected: FAIL — `assert node_info.output_schema_config is not None` fails because source nodes don't currently have `output_schema_config` set.

- [ ] **Step 3: Implement — populate source `output_schema_config`**

In `src/elspeth/core/dag/builder.py`, modify the source node creation block (around line 154). After `source_config = source.config`, parse the schema:

```python
    # Add source
    source_config = source.config
    source_schema_config = SchemaConfig.from_dict(source_config["schema"])
    source_id = node_id("source", source.name, source_config)
    graph.add_node(
        source_id,
        node_type=NodeType.SOURCE,
        plugin_name=source.name,
        config=source_config,
        output_schema=source.output_schema,  # SourceProtocol requires this
        output_schema_config=source_schema_config,
    )
```

You will need to add `SchemaConfig` to the imports at the top of the function (or at file level). Check if `SchemaConfig` is already imported — it may be imported inside `from_plugin_instances`. If not, add:

```python
from elspeth.contracts.schema import SchemaConfig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_schema_propagation.py::TestOutputSchemaConfigPropagation::test_source_node_has_output_schema_config -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/core/dag/builder.py tests/unit/core/test_dag_schema_propagation.py
git commit -m "feat(dag): populate output_schema_config for source nodes at construction time"
```

---

### Task 2: Populate `output_schema_config` for shape-preserving transforms and sinks

Shape-preserving transforms have `_output_schema_config = None` (they don't add fields). Sinks similarly have no `output_schema_config`. Both need it populated from `config["schema"]`.

**Files:**
- Modify: `src/elspeth/core/dag/builder.py:187-210` (transforms), `166-177` (sinks)
- Test: `tests/unit/core/test_dag_schema_propagation.py`

- [ ] **Step 1: Write the failing tests**

Add to `TestOutputSchemaConfigPropagation` in `tests/unit/core/test_dag_schema_propagation.py`:

```python
def test_shape_preserving_transform_has_output_schema_config(self) -> None:
    """Transforms without _output_schema_config should still get output_schema_config
    populated from config['schema'] at construction time."""
    transform = MockTransformWithoutSchemaConfig()
    source = MockSource()
    wired = WiredTransform(
        plugin=transform,  # type: ignore[arg-type]
        settings=TransformSettings(
            name="transform_0",
            plugin=transform.name,
            input="source_out",
            on_success="output",
            on_error="discard",
            options={},
        ),
    )

    graph = ExecutionGraph.from_plugin_instances(
        source=source,  # type: ignore[arg-type]
        source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
        transforms=[wired],
        sinks={"output": MockSink()},  # type: ignore[dict-item]
        aggregations={},
        gates=[],
    )

    transform_nodes = [n for n in graph.get_nodes() if n.plugin_name == "mock_transform_no_schema"]
    assert len(transform_nodes) == 1

    node_info = transform_nodes[0]
    # Previously None — now populated from config["schema"]
    assert node_info.output_schema_config is not None
    assert node_info.output_schema_config.mode == "observed"
    assert node_info.output_schema_config.guaranteed_fields == ("config_field",)


def test_sink_node_has_output_schema_config(self) -> None:
    """Sink nodes should have output_schema_config populated from config['schema']."""
    source = MockSource()

    graph = ExecutionGraph.from_plugin_instances(
        source=source,  # type: ignore[arg-type]
        source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
        transforms=[],
        sinks={"output": MockSinkWithSchema()},  # type: ignore[dict-item]
        aggregations={},
        gates=[],
    )

    sink_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.SINK]
    assert len(sink_nodes) == 1

    node_info = sink_nodes[0]
    assert node_info.output_schema_config is not None
    assert node_info.output_schema_config.mode == "observed"
```

Also add a new mock for sinks with schema (the existing `MockSink` has `config = {}` with no schema):

```python
class MockSinkWithSchema:
    """Mock sink plugin with schema config."""

    name = "mock_sink_schema"
    input_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed"}}
    _on_write_failure: str = "discard"

    def _reset_diversion_log(self) -> None:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_schema_propagation.py::TestOutputSchemaConfigPropagation::test_shape_preserving_transform_has_output_schema_config tests/unit/core/test_dag_schema_propagation.py::TestOutputSchemaConfigPropagation::test_sink_node_has_output_schema_config -v`
Expected: Both FAIL.

- [ ] **Step 3: Implement — populate transform and sink `output_schema_config`**

In `src/elspeth/core/dag/builder.py`, modify the transform loop (around line 187-210):

```python
    for seq, wired in enumerate(transforms):
        transform = wired.plugin
        transform_config = transform.config
        tid = node_id("transform", wired.settings.name, transform_config)
        transform_ids_by_name[wired.settings.name] = tid
        transform_ids_by_seq[seq] = tid

        node_config = dict(transform_config)
        node_type = NodeType.TRANSFORM

        # Validate output schema contract — crash if transform declares output
        # fields but provides no DAG contract.
        _validate_output_schema_contract(transform)
        output_schema_config = transform._output_schema_config

        # Shape-preserving transforms don't compute _output_schema_config.
        # Parse from config["schema"] so every node has a typed schema.
        if output_schema_config is None and "schema" in transform_config:
            output_schema_config = SchemaConfig.from_dict(transform_config["schema"])

        graph.add_node(
            tid,
            node_type=node_type,
            plugin_name=transform.name,
            config=node_config,
            input_schema=transform.input_schema,
            output_schema=transform.output_schema,
            output_schema_config=output_schema_config,
        )
```

Similarly, modify the sink loop (around line 165-177):

```python
    # Add sinks
    sink_ids: dict[SinkName, NodeID] = {}
    for sink_name, sink in sinks.items():
        sink_config = sink.config
        sid = node_id("sink", sink_name, sink_config)
        sink_ids[SinkName(sink_name)] = sid

        sink_schema_config: SchemaConfig | None = None
        if "schema" in sink_config:
            sink_schema_config = SchemaConfig.from_dict(sink_config["schema"])

        graph.add_node(
            sid,
            node_type=NodeType.SINK,
            plugin_name=sink.name,
            config=sink_config,
            input_schema=sink.input_schema,
            output_schema_config=sink_schema_config,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_schema_propagation.py::TestOutputSchemaConfigPropagation -v`
Expected: All tests in this class PASS.

- [ ] **Step 5: Also fix aggregation nodes (same pattern)**

In `src/elspeth/core/dag/builder.py`, modify the aggregation loop (around line 214-239):

```python
        # Same validation for aggregation transforms.
        _validate_output_schema_contract(transform)
        agg_output_schema_config = transform._output_schema_config

        # Aggregation transforms without _output_schema_config: parse from config.
        if agg_output_schema_config is None and "schema" in transform_config:
            agg_output_schema_config = SchemaConfig.from_dict(transform_config["schema"])

        graph.add_node(
            aid,
            node_type=NodeType.AGGREGATION,
            plugin_name=agg_config.plugin,
            config=agg_node_config,
            input_schema=transform.input_schema,
            output_schema=transform.output_schema,
            output_schema_config=agg_output_schema_config,
        )
```

- [ ] **Step 6: Run full test suite for this file**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_schema_propagation.py -v`
Expected: All tests PASS (existing tests should not regress).

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/core/dag/builder.py tests/unit/core/test_dag_schema_propagation.py
git commit -m "feat(dag): populate output_schema_config for all node types at construction time"
```

---

### Task 3: Simplify `_assign_schema` — drop the dict write

Now that all producer nodes have `output_schema_config`, `_assign_schema` only needs to set the typed field on pass-through nodes. The `config["schema"]` write is dead.

**Files:**
- Modify: `src/elspeth/core/dag/builder.py:140-149`
- Test: `tests/unit/core/test_dag_schema_propagation.py`

- [ ] **Step 1: Write the failing test**

Add a new test class in `tests/unit/core/test_dag_schema_propagation.py`:

```python
class TestPassThroughNodesUseTypedSchema:
    """After single-source-of-truth refactor, pass-through nodes (gates, coalesce)
    should have output_schema_config populated but should NOT have config['schema'].
    """

    def test_gate_has_output_schema_config_not_dict(self) -> None:
        """Gate should have output_schema_config but no config['schema']."""
        transform = MockTransformWithSchemaConfig()
        source = MockSource()
        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="llm_step",
                plugin=transform.name,
                input="source_out",
                on_success="gate_in",
                on_error="discard",
                options={},
            ),
        )

        gate = GateSettings(
            name="quality_gate",
            input="gate_in",
            condition="True",
            routes={"true": "output", "false": "output"},
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[gate],
        )

        gate_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.GATE]
        assert len(gate_nodes) == 1

        gate_info = gate_nodes[0]
        # Typed schema is populated
        assert gate_info.output_schema_config is not None
        assert gate_info.output_schema_config.guaranteed_fields == ("field_a", "field_b")
        assert gate_info.output_schema_config.audit_fields == ("field_c", "field_d")
        # Dict form is NOT written to pass-through nodes
        assert "schema" not in gate_info.config

    def test_coalesce_has_output_schema_config_not_dict(self) -> None:
        """Coalesce should have output_schema_config but no config['schema']."""
        transform = MockTransformWithSchemaConfig()
        source = MockSource()

        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="llm_step",
                plugin=transform.name,
                input="source_out",
                on_success="fork_in",
                on_error="discard",
                options={},
            ),
        )

        fork_gate = GateSettings(
            name="splitter",
            input="fork_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["branch_a", "branch_b"],
        )

        coalesce = CoalesceSettings(
            name="merger",
            branches=["branch_a", "branch_b"],
            policy="require_all",
            merge="union",
            on_success="output",
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[fork_gate],
            coalesce_settings=[coalesce],
        )

        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1

        coal_info = coalesce_nodes[0]
        assert coal_info.output_schema_config is not None
        assert coal_info.output_schema_config.guaranteed_fields == ("field_a", "field_b")
        assert "schema" not in coal_info.config
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_schema_propagation.py::TestPassThroughNodesUseTypedSchema -v`
Expected: Both FAIL — gates/coalesce still have `config["schema"]` because `_assign_schema` still writes it.

- [ ] **Step 3: Implement — simplify `_assign_schema`**

In `src/elspeth/core/dag/builder.py`, replace the `_assign_schema` function (around line 140-149):

```python
    def _assign_schema(target_nid: NodeID, schema: SchemaConfig) -> None:
        """Set output_schema_config on a pass-through node (gate or coalesce).

        Pass-through nodes don't have their own schema — they inherit from
        upstream producers. This sets the typed SchemaConfig so all consumers
        can read it directly without fallback chains.
        """
        target_info = graph.get_node_info(target_nid)
        object.__setattr__(target_info, "output_schema_config", schema)
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_schema_propagation.py::TestPassThroughNodesUseTypedSchema -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/core/dag/builder.py tests/unit/core/test_dag_schema_propagation.py
git commit -m "refactor(dag): _assign_schema only sets output_schema_config, drops dict write"
```

---

### Task 4: Simplify `_best_schema_config` and gate schema guard

Now that all producer nodes have `output_schema_config`, the fallback to `config["schema"]` is dead code.

**Files:**
- Modify: `src/elspeth/core/dag/builder.py:126-138` (`_best_schema_config`), line 564 (gate guard)

- [ ] **Step 1: Simplify `_best_schema_config`**

Replace the function (around line 126-138):

```python
    def _best_schema_config(nid: NodeID) -> SchemaConfig:
        """Get SchemaConfig from a node.

        All nodes have output_schema_config populated at construction time
        (sources, transforms, aggregations from config; gates and coalesce
        from upstream inheritance via _assign_schema).
        """
        info = graph.get_node_info(nid)
        if info.output_schema_config is None:
            raise FrameworkBugError(
                f"Node '{nid}' has no output_schema_config. "
                "All producer nodes must have output_schema_config populated "
                "at construction time."
            )
        return info.output_schema_config
```

Check if `FrameworkBugError` is imported. Search the file for existing uses — it's likely already imported. If not, add:

```python
from elspeth.contracts.errors import FrameworkBugError
```

- [ ] **Step 2: Simplify the gate schema guard**

In the gate schema resolution (pass 1), around line 564, replace:

```python
        if upstream_info.output_schema_config is not None or "schema" in upstream_info.config:
```

with:

```python
        if upstream_info.output_schema_config is not None:
```

- [ ] **Step 3: Run the full schema propagation test suite**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_schema_propagation.py -v`
Expected: All passing tests still pass. Some old tests in `TestPassThroughNodesInheritComputedSchema` will now FAIL because they assert `config["schema"]` on gates/coalesces — we'll fix those in Task 6.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/core/dag/builder.py
git commit -m "refactor(dag): simplify _best_schema_config — no dict fallback"
```

---

### Task 5: Collapse `get_schema_config_from_node` and orchestrator fallbacks

**Files:**
- Modify: `src/elspeth/core/dag/graph.py:1428-1466`
- Modify: `src/elspeth/engine/orchestrator/core.py:1324-1331`

- [ ] **Step 1: Simplify `get_schema_config_from_node`**

In `src/elspeth/core/dag/graph.py`, replace the method (around line 1428-1466):

```python
    def get_schema_config_from_node(self, node_id: str) -> SchemaConfig | None:
        """Extract SchemaConfig from node.

        Returns output_schema_config directly — all nodes with schemas have
        this populated at construction time by the builder.

        Args:
            node_id: Node ID to get schema config from

        Returns:
            SchemaConfig if available, None if node has no schema
        """
        node_info = self.get_node_info(node_id)
        return node_info.output_schema_config
```

Remove the `Mapping` import if it was only used by the deleted `isinstance` check. Check other uses of `Mapping` in the file first — it's likely used elsewhere so keep it.

- [ ] **Step 2: Simplify orchestrator schema resolution**

In `src/elspeth/engine/orchestrator/core.py`, replace the fallback block (around line 1324-1331):

```python
            # Schema config is always available via output_schema_config —
            # populated at construction time for all node types.
            schema_config = node_info.output_schema_config
            if schema_config is None:
                raise FrameworkBugError(
                    f"Node '{node_id}' has no output_schema_config. "
                    "All nodes in execution order must have schema config "
                    "populated by the builder."
                )
```

Check if `FrameworkBugError` is already imported in this file. If not, add:

```python
from elspeth.contracts.errors import FrameworkBugError
```

Also remove the now-unused `SchemaConfig` import if it was only used in the fallback. Check — it's likely used elsewhere in the file (e.g., the `from elspeth.contracts.schema import SchemaConfig` at line 1296 is inside the function). If `SchemaConfig` is only used in the deleted `SchemaConfig.from_dict()` call, remove it. If used elsewhere, keep it.

- [ ] **Step 3: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/unit/core/ tests/unit/engine/ tests/integration/core/ -x -v --tb=short`
Expected: Some tests in `test_dag_schema_propagation.py` may fail (we fix those in Task 6). Everything else should pass.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/core/dag/graph.py src/elspeth/engine/orchestrator/core.py
git commit -m "refactor(dag,orchestrator): collapse schema fallback chains to direct reads"
```

---

### Task 6: Update tests — remove dead fallback/aliasing tests, fix assertions

Several existing tests assert the old behavior (fallback chains, `config["schema"]` on pass-through nodes, dict aliasing). These must be updated to match the new single-source-of-truth design.

**Files:**
- Modify: `tests/unit/core/test_dag_schema_propagation.py`

- [ ] **Step 1: Update `test_transform_without_schema_config_has_none`**

This test (line 130) asserts `output_schema_config is None` for shape-preserving transforms. After Task 2, they always have it. **Replace the test:**

```python
    def test_transform_without_plugin_schema_gets_config_schema(self) -> None:
        """Transforms without _output_schema_config get output_schema_config
        populated from config['schema'] at construction time."""
        transform = MockTransformWithoutSchemaConfig()
        source = MockSource()
        wired = WiredTransform(
            plugin=transform,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="transform_0",
                plugin=transform.name,
                input="source_out",
                on_success="output",
                on_error="discard",
                options={},
            ),
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired],
            sinks={"output": MockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[],
        )

        transform_nodes = [n for n in graph.get_nodes() if n.plugin_name == "mock_transform_no_schema"]
        assert len(transform_nodes) == 1

        node_info = transform_nodes[0]
        # Now populated from config["schema"]
        assert node_info.output_schema_config is not None
        assert node_info.output_schema_config.mode == "observed"
        assert node_info.output_schema_config.guaranteed_fields == ("config_field",)
```

- [ ] **Step 2: Update `TestGetSchemaConfigFromNodePriority`**

**Replace `test_falls_back_to_config_dict_when_no_nodeinfo_schema` (line 198)** — the fallback no longer exists. Replace with a test that verifies `get_schema_config_from_node` returns the typed schema directly:

```python
    def test_returns_output_schema_config_directly(self) -> None:
        """get_schema_config_from_node returns output_schema_config without parsing config dict."""
        graph = ExecutionGraph()

        schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("field_a",),
        )

        graph.add_node(
            "test_node",
            node_type=NodeType.TRANSFORM,
            plugin_name="test",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["different_field"]}},
            output_schema_config=schema,
        )

        result = graph.get_schema_config_from_node("test_node")
        # Returns the typed object, ignores config dict
        assert result is schema
        assert result.guaranteed_fields == ("field_a",)
```

Keep `test_returns_none_when_no_schema_anywhere` (line 219) — it still applies (sinks without schema config have `output_schema_config = None`).

- [ ] **Step 3: Update `TestPassThroughNodesInheritComputedSchema`**

Replace all assertions on `config["schema"]` with assertions on `output_schema_config`. The four tests in this class (lines 538-751) should be updated. For each test, change the assertion pattern from:

```python
gate_schema_dict = gate_nodes[0].config["schema"]
assert "guaranteed_fields" in gate_schema_dict
assert set(gate_schema_dict["guaranteed_fields"]) == {"field_a", "field_b"}
```

to:

```python
gate_schema = gate_nodes[0].output_schema_config
assert gate_schema is not None
assert set(gate_schema.guaranteed_fields) == {"field_a", "field_b"}
assert set(gate_schema.audit_fields) == {"field_c", "field_d"}
```

Apply this pattern to all four tests:
- `test_gate_config_schema_includes_computed_guaranteed_fields` → rename to `test_gate_inherits_computed_schema_config`
- `test_gate_config_schema_falls_back_to_raw_when_no_computed` → rename to `test_gate_inherits_raw_schema_when_no_computed`
- `test_coalesce_config_schema_includes_computed_fields` → rename to `test_coalesce_inherits_computed_schema_config`
- `test_deferred_gate_after_coalesce_inherits_computed_schema` → keep name, update assertions

For `test_gate_config_schema_falls_back_to_raw_when_no_computed` (line 586), the assertion changes from:

```python
gate_schema_dict = gate_nodes[0].config["schema"]
assert gate_schema_dict["guaranteed_fields"] == ("config_field",)
```

to:

```python
gate_schema = gate_nodes[0].output_schema_config
assert gate_schema is not None
assert gate_schema.guaranteed_fields == ("config_field",)
```

- [ ] **Step 4: Delete `TestSchemaAliasingPrevention`**

Delete the entire `TestSchemaAliasingPrevention` class (lines 754-861). These tests verified that `config["schema"]` dicts were independent objects to prevent mutation aliasing. Since pass-through nodes no longer have `config["schema"]`, and `SchemaConfig` is a frozen dataclass (immutable), aliasing is safe by construction. No replacement tests needed.

- [ ] **Step 5: Run the full schema propagation test suite**

Run: `.venv/bin/python -m pytest tests/unit/core/test_dag_schema_propagation.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run broader test suites to catch regressions**

Run: `.venv/bin/python -m pytest tests/unit/core/ tests/integration/core/ -v --tb=short`
Expected: ALL PASS. If anything fails, it's likely a test that reads `config["schema"]` on a gate/coalesce node — fix by reading `output_schema_config` instead.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/core/test_dag_schema_propagation.py
git commit -m "test(dag): update schema propagation tests for single source of truth"
```

---

### Task 7: Full regression — run all tests, mypy, ruff

**Files:** None (verification only)

- [ ] **Step 1: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/core/dag/builder.py src/elspeth/core/dag/graph.py src/elspeth/engine/orchestrator/core.py`
Expected: No new errors. If there are type errors from removed imports or changed return types, fix them.

- [ ] **Step 2: Run ruff**

Run: `.venv/bin/python -m ruff check src/elspeth/core/dag/builder.py src/elspeth/core/dag/graph.py src/elspeth/engine/orchestrator/core.py`
Expected: No new lint errors. Fix any unused imports flagged by ruff.

- [ ] **Step 3: Run tier model enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: PASS. The `SchemaConfig` import in builder.py is from `contracts/` (L0) into `core/` (L1), which is a valid downward import.

- [ ] **Step 4: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -x --tb=short`
Expected: ALL PASS. Watch for:
- Integration tests that assert `config["schema"]` on gate/coalesce nodes
- Tests that use `get_schema_config_from_node()` expecting fallback behavior

If any fail, fix the assertions (read `output_schema_config` instead of `config["schema"]`).

- [ ] **Step 5: Commit any fixups**

```bash
git add -u
git commit -m "fix: address lint/type/test regressions from schema single-source-of-truth"
```

---

## Summary of Changes

| Before | After |
|--------|-------|
| `_assign_schema` writes BOTH `config["schema"]` and `output_schema_config` | `_assign_schema` writes only `output_schema_config` |
| `_best_schema_config` falls back from typed → dict | `_best_schema_config` reads typed directly (crash on None) |
| `get_schema_config_from_node` falls back from typed → dict → None | `get_schema_config_from_node` returns `output_schema_config` directly |
| Orchestrator falls back from typed → dict | Orchestrator reads typed directly (crash on None) |
| Source nodes: `output_schema_config = None` | Source nodes: `output_schema_config` parsed from config |
| Shape-preserving transforms: `output_schema_config = None` | Shape-preserving transforms: `output_schema_config` parsed from config |
| Sinks: `output_schema_config = None` | Sinks: `output_schema_config` parsed from config (or None if no schema) |
| Gates/coalesce: `config["schema"]` written by `_assign_schema` | Gates/coalesce: NO `config["schema"]` — only typed `output_schema_config` |
| 4 fallback chains (builder×2, graph×1, orchestrator×1) | 0 fallback chains |
| `TestSchemaAliasingPrevention` (mutable dict aliasing risk) | Deleted — frozen `SchemaConfig` is aliasing-safe by construction |

## What's NOT Changed

- `config["schema"]` on sources, transforms, sinks, and aggregations — it's part of the YAML config and node ID hash. We only stopped **reading** it as a schema source.
- `validation.py:197` — reads `config["schema"]` for error reporting on plugin config validation (runs before graph build, only touches source/transform/sink configs).
- `builder.py:222` — includes `"schema"` in aggregation node config for ID hashing. This is correct and unchanged.
