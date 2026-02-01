# ⚠️ SUPERSEDED - DO NOT IMPLEMENT ⚠️

**This plan has been superseded by a root cause fix.**

**Status:** ❌ **OBSOLETE**
**Superseded by:** `docs/plans/2026-01-24-fix-schema-validation-properly.md`
**Date Superseded:** 2026-01-24

**Why Superseded:**

This plan (Option A - SchemaConfig propagation) would improve architecture from 2/5 to 4/5 by eliminating Pydantic introspection. However, **it doesn't fix the root cause** - validation happening at the wrong architectural layer.

**The proper fix** moves schema validation to plugin construction time, where SchemaConfig already exists. This:
- Achieves 5/5 architecture (not 4/5)
- Eliminates 200+ lines of DAG validation code (not adds NodeInfo fields)
- Fixes information loss pattern (not works around it)
- Pre-release window allows breaking changes (post-release would require migration)

**See root cause fix plan:** `docs/plans/2026-01-24-fix-schema-validation-properly.md`

---

# ~~Eliminate Parallel Schema Detection Implementation Plan~~ (OBSOLETE)

> ~~**For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.~~

**⚠️ DO NOT EXECUTE THIS PLAN - Use the proper fix plan instead**

**~~Goal:~~** ~~Replace Pydantic introspection-based dynamic schema detection with SchemaConfig propagation to DAG nodes.~~

**Architecture:** Extend `NodeInfo` to store `SchemaConfig` metadata alongside Pydantic schemas. Extract `SchemaConfig` from plugin config during graph construction. Update validation logic to use `SchemaConfig.is_dynamic` (single source of truth) instead of introspecting `model_fields` and `model_config`.

**Tech Stack:** Python dataclasses, Pydantic models, NetworkX DAG

**Benefits:**
- Eliminates redundant detection mechanisms (2 → 1)
- Removes Pydantic coupling from validation logic
- Single source of truth (`SchemaConfig.is_dynamic`)
- Improves architecture quality from 2/5 to 4-5/5

**Related Bugs:**
- P0-2026-01-24-eliminate-parallel-dynamic-schema-detection (this implements the fix)
- P0-2026-01-24-dynamic-schema-detection-regression (parent issue that created tech debt)

---

## Task 1: Extend NodeInfo Dataclass

**Goal:** Add optional `SchemaConfig` fields to `NodeInfo` for input and output schemas.

**Files:**
- Modify: `src/elspeth/core/dag.py:32-46` (NodeInfo dataclass)

**Step 1: Write failing test for NodeInfo extension**

Add to `tests/core/test_dag.py` at end of file:

```python
class TestSchemaConfigPropagation:
    """Test that SchemaConfig metadata is propagated to DAG nodes."""

    def test_node_info_stores_schema_config(self) -> None:
        """NodeInfo should store SchemaConfig alongside Pydantic schemas."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.dag import NodeInfo

        schema_config = SchemaConfig.from_dict({"fields": "dynamic"})

        node = NodeInfo(
            node_id="test_node",
            node_type="transform",
            plugin_name="passthrough",
            config={},
            input_schema=None,
            output_schema=None,
            input_schema_config=schema_config,
            output_schema_config=schema_config,
        )

        assert node.input_schema_config is not None
        assert node.output_schema_config is not None
        assert node.input_schema_config.is_dynamic is True
        assert node.output_schema_config.is_dynamic is True
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_node_info_stores_schema_config -xvs
```

Expected: FAIL with "TypeError: __init__() got unexpected keyword argument 'input_schema_config'"

**Step 3: Extend NodeInfo dataclass**

In `src/elspeth/core/dag.py:32-46`, modify the `NodeInfo` dataclass:

```python
@dataclass
class NodeInfo:
    """Information about a node in the execution graph.

    Schemas are immutable after graph construction. Even dynamic schemas
    (determined by data inspection) are locked at launch and never change
    during the run. This guarantees audit trail consistency.

    Schema metadata (SchemaConfig) is stored alongside Pydantic schemas
    to provide access to is_dynamic flag without introspecting Pydantic internals.
    """

    node_id: str
    node_type: str  # source, transform, gate, aggregation, coalesce, sink
    plugin_name: str
    config: dict[str, Any] = field(default_factory=dict)
    input_schema: type[PluginSchema] | None = None  # Immutable after graph construction
    output_schema: type[PluginSchema] | None = None  # Immutable after graph construction
    input_schema_config: SchemaConfig | None = None  # Metadata for input schema
    output_schema_config: SchemaConfig | None = None  # Metadata for output schema
```

**Step 4: Add import for SchemaConfig**

At top of `src/elspeth/core/dag.py`, add to imports section (around line 20):

```python
if TYPE_CHECKING:
    from elspeth.contracts import PluginSchema
    from elspeth.contracts.schema import SchemaConfig  # NEW
    from elspeth.core.config import AggregationSettings, CoalesceSettings, GateSettings
    from elspeth.plugins.protocols import SinkProtocol, SourceProtocol, TransformProtocol
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_node_info_stores_schema_config -xvs
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat: add SchemaConfig fields to NodeInfo

- Add input_schema_config and output_schema_config optional fields
- Enables validation to use SchemaConfig.is_dynamic instead of introspection
- Part of eliminating parallel dynamic schema detection mechanisms
- Ref: P0-2026-01-24-eliminate-parallel-dynamic-schema-detection"
```

---

## Task 2: Create Helper for SchemaConfig Extraction

**Goal:** Create reusable helper to extract `SchemaConfig` from plugin options dict.

**Files:**
- Modify: `src/elspeth/core/dag.py` (add helper before `from_plugin_instances`)

**Step 1: Write failing test for extraction helper**

Add to `tests/core/test_dag.py` in `TestSchemaConfigPropagation` class:

```python
def test_extract_schema_config_from_options(self) -> None:
    """Helper should extract SchemaConfig from plugin options dict."""
    from elspeth.core.dag import _extract_schema_config

    # Dynamic schema
    options = {"schema": {"fields": "dynamic"}}
    config = _extract_schema_config(options)
    assert config.is_dynamic is True

    # Explicit schema
    options = {"schema": {"mode": "strict", "fields": ["id: int", "name: str"]}}
    config = _extract_schema_config(options)
    assert config.is_dynamic is False
    assert len(config.fields) == 2

    # Missing schema key
    options = {}
    config = _extract_schema_config(options)
    assert config is None

    # Empty schema dict
    options = {"schema": {}}
    config = _extract_schema_config(options)
    assert config is None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_extract_schema_config_from_options -xvs
```

Expected: FAIL with "ImportError: cannot import name '_extract_schema_config'"

**Step 3: Implement extraction helper**

In `src/elspeth/core/dag.py`, add this function before `class ExecutionGraph` (around line 100):

```python
def _extract_schema_config(options: dict[str, Any]) -> SchemaConfig | None:
    """Extract SchemaConfig from plugin options dict.

    Args:
        options: Plugin config options dict (may contain 'schema' key)

    Returns:
        SchemaConfig instance if schema key exists, None otherwise

    Note:
        Returns None if schema key is missing or if schema dict is empty.
        This handles plugins that don't define schemas.
    """
    from elspeth.contracts.schema import SchemaConfig

    schema_dict = options.get("schema")
    if not schema_dict:
        return None

    # Handle empty schema dict (plugin with no schema defined)
    if isinstance(schema_dict, dict) and not schema_dict:
        return None

    return SchemaConfig.from_dict(schema_dict)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_extract_schema_config_from_options -xvs
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat: add _extract_schema_config helper

- Extracts SchemaConfig from plugin options dict
- Returns None if schema key missing or empty
- Centralizes extraction logic for reuse in graph construction
- Ref: P0-2026-01-24-eliminate-parallel-dynamic-schema-detection"
```

---

## Task 3: Update from_plugin_instances - Sources

**Goal:** Extract SchemaConfig from source and store in NodeInfo.

**Files:**
- Modify: `src/elspeth/core/dag.py:528-536` (source node construction)

**Step 1: Write failing test for source SchemaConfig**

Add to `tests/core/test_dag.py` in `TestSchemaConfigPropagation` class:

```python
def test_source_node_has_schema_config(self) -> None:
    """Source nodes should store SchemaConfig from instance."""
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.dag import ExecutionGraph
    from elspeth.plugins.sources.csv_source import CSVSource

    # Create source with dynamic schema
    source_config = {
        "path": "test.csv",
        "schema": {"fields": "dynamic"},
    }
    source = CSVSource(source_config)

    graph = ExecutionGraph.from_plugin_instances(
        source=source,
        transforms=[],
        sinks={},
        aggregations={},
        gates=[],
        output_sink="",
        coalesce_settings=[],
    )

    # Get source node
    source_nodes = [n for n, data in graph._graph.nodes(data=True) if data["info"].node_type == "source"]
    assert len(source_nodes) == 1

    source_info = graph.get_node_info(source_nodes[0])
    assert source_info.output_schema_config is not None
    assert source_info.output_schema_config.is_dynamic is True
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_source_node_has_schema_config -xvs
```

Expected: FAIL with "AssertionError: assert None is not None" (output_schema_config not set)

**Step 3: Update source node construction**

In `src/elspeth/core/dag.py:528-536`, modify source node creation:

```python
# Add source - extract schema from instance
source_id = node_id("source", source.name)

# Extract SchemaConfig from source config
source_schema_config = getattr(source, "_schema_config", None)

graph.add_node(
    source_id,
    node_type="source",
    plugin_name=source.name,
    config={},
    output_schema=getattr(source, "output_schema", None),
    output_schema_config=source_schema_config,  # NEW
)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_source_node_has_schema_config -xvs
```

Expected: PASS

**Step 5: Run full test suite to check for regressions**

```bash
pytest tests/core/test_dag.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat: propagate SchemaConfig from source to graph

- Extract _schema_config from source instance
- Store in output_schema_config field of source node
- Enables validation to access is_dynamic without introspection
- Ref: P0-2026-01-24-eliminate-parallel-dynamic-schema-detection"
```

---

## Task 4: Update from_plugin_instances - Transforms

**Goal:** Extract SchemaConfig from transforms and store in NodeInfo.

**Files:**
- Modify: `src/elspeth/core/dag.py:558-572` (transform node construction)

**Step 1: Write failing test for transform SchemaConfig**

Add to `tests/core/test_dag.py` in `TestSchemaConfigPropagation` class:

```python
def test_transform_node_has_schema_config(self) -> None:
    """Transform nodes should store SchemaConfig from instance."""
    from elspeth.core.dag import ExecutionGraph
    from elspeth.plugins.sources.csv_source import CSVSource
    from elspeth.plugins.transforms.passthrough import PassThrough

    # Create source and transform with dynamic schemas
    source = CSVSource({"path": "test.csv", "schema": {"fields": "dynamic"}})
    transform = PassThrough({"schema": {"fields": "dynamic"}})

    graph = ExecutionGraph.from_plugin_instances(
        source=source,
        transforms=[transform],
        sinks={},
        aggregations={},
        gates=[],
        output_sink="",
        coalesce_settings=[],
    )

    # Get transform node
    transform_nodes = [n for n, data in graph._graph.nodes(data=True) if data["info"].node_type == "transform"]
    assert len(transform_nodes) == 1

    transform_info = graph.get_node_info(transform_nodes[0])
    assert transform_info.input_schema_config is not None
    assert transform_info.output_schema_config is not None
    assert transform_info.input_schema_config.is_dynamic is True
    assert transform_info.output_schema_config.is_dynamic is True
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_transform_node_has_schema_config -xvs
```

Expected: FAIL with "AssertionError: assert None is not None" (schema_configs not set)

**Step 3: Update transform node construction**

In `src/elspeth/core/dag.py:558-572`, modify transform loop:

```python
for i, transform in enumerate(transforms):
    tid = node_id("transform", transform.name)
    transform_ids[i] = tid

    # Extract SchemaConfig from transform config
    transform_schema_config = getattr(transform, "_schema_config", None)

    graph.add_node(
        tid,
        node_type="transform",
        plugin_name=transform.name,
        config={},
        input_schema=getattr(transform, "input_schema", None),
        output_schema=getattr(transform, "output_schema", None),
        input_schema_config=transform_schema_config,   # NEW
        output_schema_config=transform_schema_config,  # NEW - same config for both
    )

    graph.add_edge(prev_node_id, tid, label="continue", mode=RoutingMode.MOVE)
    prev_node_id = tid
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_transform_node_has_schema_config -xvs
```

Expected: PASS

**Step 5: Run full test suite**

```bash
pytest tests/core/test_dag.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat: propagate SchemaConfig from transforms to graph

- Extract _schema_config from transform instances
- Store in input_schema_config and output_schema_config
- Transforms use same SchemaConfig for both input and output
- Ref: P0-2026-01-24-eliminate-parallel-dynamic-schema-detection"
```

---

## Task 5: Update from_plugin_instances - Sinks

**Goal:** Extract SchemaConfig from sinks and store in NodeInfo.

**Files:**
- Modify: `src/elspeth/core/dag.py:538-549` (sink node construction)

**Step 1: Write failing test for sink SchemaConfig**

Add to `tests/core/test_dag.py` in `TestSchemaConfigPropagation` class:

```python
def test_sink_node_has_schema_config(self) -> None:
    """Sink nodes should store SchemaConfig from instance."""
    from elspeth.core.dag import ExecutionGraph
    from elspeth.plugins.sinks.csv_sink import CSVSink
    from elspeth.plugins.sources.csv_source import CSVSource

    # Create source and sink
    source = CSVSource({"path": "test.csv", "schema": {"fields": "dynamic"}})
    sink = CSVSink({"path": "output.csv", "schema": {"fields": "dynamic"}})

    graph = ExecutionGraph.from_plugin_instances(
        source=source,
        transforms=[],
        sinks={"output": sink},
        aggregations={},
        gates=[],
        output_sink="output",
        coalesce_settings=[],
    )

    # Get sink node
    sink_nodes = [n for n, data in graph._graph.nodes(data=True) if data["info"].node_type == "sink"]
    assert len(sink_nodes) == 1

    sink_info = graph.get_node_info(sink_nodes[0])
    assert sink_info.input_schema_config is not None
    assert sink_info.input_schema_config.is_dynamic is True
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_sink_node_has_schema_config -xvs
```

Expected: FAIL with "AssertionError: assert None is not None"

**Step 3: Update sink node construction**

In `src/elspeth/core/dag.py:538-549`, modify sink loop:

```python
# Add sinks
sink_ids: dict[str, str] = {}
for sink_name, sink in sinks.items():
    sid = node_id("sink", sink_name)
    sink_ids[sink_name] = sid

    # Extract SchemaConfig from sink config
    sink_schema_config = getattr(sink, "_schema_config", None)

    graph.add_node(
        sid,
        node_type="sink",
        plugin_name=sink.name,
        config={},
        input_schema=getattr(sink, "input_schema", None),
        input_schema_config=sink_schema_config,  # NEW
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_sink_node_has_schema_config -xvs
```

Expected: PASS

**Step 5: Run full test suite**

```bash
pytest tests/core/test_dag.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat: propagate SchemaConfig from sinks to graph

- Extract _schema_config from sink instances
- Store in input_schema_config field of sink nodes
- Ref: P0-2026-01-24-eliminate-parallel-dynamic-schema-detection"
```

---

## Task 6: Update from_plugin_instances - Aggregations

**Goal:** Extract SchemaConfig from aggregations and store in NodeInfo.

**Files:**
- Modify: `src/elspeth/core/dag.py:576-598` (aggregation node construction)

**Step 1: Write failing test for aggregation SchemaConfig**

Add to `tests/core/test_dag.py` in `TestSchemaConfigPropagation` class:

```python
def test_aggregation_node_has_schema_config(self) -> None:
    """Aggregation nodes should store SchemaConfig from instance."""
    from elspeth.core.config import AggregationSettings, TriggerConfig
    from elspeth.core.dag import ExecutionGraph
    from elspeth.plugins.sources.csv_source import CSVSource
    from elspeth.plugins.transforms.batch_stats import BatchStats

    # Create source and aggregation
    source = CSVSource({"path": "test.csv", "schema": {"fields": "dynamic"}})
    agg_transform = BatchStats({
        "schema": {"fields": "dynamic"},
        "trigger": {"type": "count", "count": 10},
    })
    agg_settings = AggregationSettings(
        plugin="batch_stats",
        trigger=TriggerConfig(type="count", count=10),
        output_mode="single",
        options={},
    )

    graph = ExecutionGraph.from_plugin_instances(
        source=source,
        transforms=[],
        sinks={},
        aggregations={"batch": (agg_transform, agg_settings)},
        gates=[],
        output_sink="",
        coalesce_settings=[],
    )

    # Get aggregation node
    agg_nodes = [n for n, data in graph._graph.nodes(data=True) if data["info"].node_type == "aggregation"]
    assert len(agg_nodes) == 1

    agg_info = graph.get_node_info(agg_nodes[0])
    assert agg_info.input_schema_config is not None
    assert agg_info.output_schema_config is not None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_aggregation_node_has_schema_config -xvs
```

Expected: FAIL with "AssertionError: assert None is not None"

**Step 3: Update aggregation node construction**

In `src/elspeth/core/dag.py:576-598`, modify aggregation loop:

```python
# Build aggregations - dual schemas
aggregation_ids: dict[str, str] = {}
for agg_name, (transform, agg_config) in aggregations.items():
    aid = node_id("aggregation", agg_name)
    aggregation_ids[agg_name] = aid

    agg_node_config = {
        "trigger": agg_config.trigger.model_dump(),
        "output_mode": agg_config.output_mode,
        "options": dict(agg_config.options),
    }

    # Extract SchemaConfig from aggregation transform
    agg_schema_config = getattr(transform, "_schema_config", None)

    graph.add_node(
        aid,
        node_type="aggregation",
        plugin_name=agg_config.plugin,
        config=agg_node_config,
        input_schema=getattr(transform, "input_schema", None),
        output_schema=getattr(transform, "output_schema", None),
        input_schema_config=agg_schema_config,   # NEW
        output_schema_config=agg_schema_config,  # NEW
    )

    graph.add_edge(prev_node_id, aid, label="continue", mode=RoutingMode.MOVE)
    prev_node_id = aid
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_aggregation_node_has_schema_config -xvs
```

Expected: PASS

**Step 5: Run full test suite**

```bash
pytest tests/core/test_dag.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat: propagate SchemaConfig from aggregations to graph

- Extract _schema_config from aggregation transform instances
- Store in input_schema_config and output_schema_config
- Ref: P0-2026-01-24-eliminate-parallel-dynamic-schema-detection"
```

---

## Task 7: Update Edge Schema Validation

**Goal:** Replace `_is_dynamic_schema()` introspection with `SchemaConfig.is_dynamic` check.

**Files:**
- Modify: `src/elspeth/core/dag.py:288-338` (_validate_edge_schemas method)

**Step 1: Write failing test for new validation logic**

Add to `tests/core/test_dag.py` in `TestSchemaConfigPropagation` class:

```python
def test_validation_uses_schema_config_not_introspection(self) -> None:
    """Validation should check SchemaConfig.is_dynamic, not introspect Pydantic."""
    from elspeth.core.dag import ExecutionGraph
    from elspeth.plugins.sources.csv_source import CSVSource
    from elspeth.plugins.sinks.csv_sink import CSVSink

    # Dynamic source -> Explicit sink (should skip validation)
    source = CSVSource({"path": "test.csv", "schema": {"fields": "dynamic"}})
    sink = CSVSink({
        "path": "output.csv",
        "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
    })

    graph = ExecutionGraph.from_plugin_instances(
        source=source,
        transforms=[],
        sinks={"output": sink},
        aggregations={},
        gates=[],
        output_sink="output",
        coalesce_settings=[],
    )

    # Add edge from source to sink
    source_nodes = [n for n, data in graph._graph.nodes(data=True) if data["info"].node_type == "source"]
    sink_nodes = [n for n, data in graph._graph.nodes(data=True) if data["info"].node_type == "sink"]
    graph.add_edge(source_nodes[0], sink_nodes[0], label="continue")

    # Should NOT raise - dynamic schema skips validation
    errors = graph.validate()
    assert len(errors) == 0
```

**Step 2: Run test to verify current behavior**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_validation_uses_schema_config_not_introspection -xvs
```

Expected: PASS (test passes with current introspection approach)

**Step 3: Update _validate_edge_schemas to use SchemaConfig**

In `src/elspeth/core/dag.py:288-338`, replace the method:

```python
def _validate_edge_schemas(self) -> list[str]:
    """Validate schema compatibility along all edges.

    For each edge (producer -> consumer):
    - Get producer's effective output schema (walks through gates)
    - Get consumer's input schema
    - Check producer provides all required fields

    Dynamic schema detection now uses SchemaConfig.is_dynamic instead of
    introspecting Pydantic model structure. This eliminates coupling to
    Pydantic internals and uses the authoritative source of truth.

    Additionally validates coalesce nodes:
    - All incoming branches must have compatible schemas

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for edge in self.get_edges():
        from_info = self.get_node_info(edge.from_node)
        to_info = self.get_node_info(edge.to_node)

        # Get effective producer schema (handles gates as pass-through)
        producer_schema = self._get_effective_producer_schema(edge.from_node)

        # Get consumer input schema directly
        consumer_schema = to_info.input_schema

        # Check if either schema is dynamic using SchemaConfig (source of truth)
        producer_is_dynamic = (
            from_info.output_schema_config is not None
            and from_info.output_schema_config.is_dynamic
        )
        consumer_is_dynamic = (
            to_info.input_schema_config is not None
            and to_info.input_schema_config.is_dynamic
        )

        # Skip validation if either schema is dynamic
        if producer_is_dynamic or consumer_is_dynamic:
            continue

        # Skip if either schema is None (backwards compatibility)
        if producer_schema is None or consumer_schema is None:
            continue

        # Validate compatibility
        missing = _get_missing_required_fields(
            producer=producer_schema,
            consumer=consumer_schema,
        )

        if missing:
            errors.append(
                f"{from_info.plugin_name} -> {to_info.plugin_name} (route: {edge.label}): producer missing required fields {missing}"
            )

    # Validate coalesce nodes have compatible incoming schemas
    coalesce_nodes = [node_id for node_id, data in self._graph.nodes(data=True) if data["info"].node_type == "coalesce"]
    for coalesce_id in coalesce_nodes:
        errors.extend(self._validate_coalesce_schema_compatibility(coalesce_id))

    return errors
```

**Step 4: Run test to verify it still passes**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_validation_uses_schema_config_not_introspection -xvs
```

Expected: PASS

**Step 5: Run all DAG tests**

```bash
pytest tests/core/test_dag.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat: use SchemaConfig.is_dynamic in edge validation

- Replace _is_dynamic_schema() introspection with SchemaConfig check
- Use from_info.output_schema_config.is_dynamic (source of truth)
- Eliminates Pydantic coupling from validation logic
- Ref: P0-2026-01-24-eliminate-parallel-dynamic-schema-detection"
```

---

## Task 8: Update Coalesce Schema Validation

**Goal:** Replace `_is_dynamic_schema()` with SchemaConfig check in coalesce validation.

**Files:**
- Modify: `src/elspeth/core/dag.py:240-286` (_validate_coalesce_schema_compatibility method)

**Step 1: Write failing test for coalesce validation**

Add to `tests/core/test_dag.py` in `TestSchemaConfigPropagation` class:

```python
def test_coalesce_validation_uses_schema_config(self) -> None:
    """Coalesce validation should use SchemaConfig.is_dynamic."""
    from elspeth.contracts import PluginSchema
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.dag import ExecutionGraph

    class OutputSchema(PluginSchema):
        id: int

    # Create graph with coalesce node
    graph = ExecutionGraph()

    # Create dynamic schema config
    dynamic_config = SchemaConfig.from_dict({"fields": "dynamic"})

    # Add source with dynamic output
    graph.add_node(
        "source",
        node_type="source",
        plugin_name="csv",
        output_schema=OutputSchema,
        output_schema_config=dynamic_config,  # Dynamic via SchemaConfig
    )

    # Add two transforms (fork paths)
    graph.add_node(
        "transform1",
        node_type="transform",
        plugin_name="passthrough",
        input_schema=OutputSchema,
        output_schema=OutputSchema,
        input_schema_config=dynamic_config,
        output_schema_config=dynamic_config,
    )
    graph.add_node(
        "transform2",
        node_type="transform",
        plugin_name="passthrough",
        input_schema=OutputSchema,
        output_schema=OutputSchema,
        input_schema_config=dynamic_config,
        output_schema_config=dynamic_config,
    )

    # Add coalesce node
    graph.add_node(
        "coalesce",
        node_type="coalesce",
        plugin_name="coalesce",
        input_schema=OutputSchema,
        input_schema_config=dynamic_config,
    )

    # Connect: source forks to transform1 and transform2, then coalesce
    graph.add_edge("source", "transform1", label="path1")
    graph.add_edge("source", "transform2", label="path2")
    graph.add_edge("transform1", "coalesce", label="merge")
    graph.add_edge("transform2", "coalesce", label="merge")

    # Should NOT raise - dynamic schemas skip compatibility check
    errors = graph._validate_coalesce_schema_compatibility("coalesce")
    assert len(errors) == 0
```

**Step 2: Run test to verify current behavior**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_coalesce_validation_uses_schema_config -xvs
```

Expected: PASS (works with current introspection)

**Step 3: Update _validate_coalesce_schema_compatibility**

In `src/elspeth/core/dag.py:240-286`, replace the method:

```python
def _validate_coalesce_schema_compatibility(
    self,
    coalesce_id: str,
    raise_on_error: bool = False,
) -> list[str]:
    """Validate that all branches merging at a coalesce have compatible schemas.

    Dynamic schema detection now uses SchemaConfig.is_dynamic instead of
    introspecting Pydantic model structure.

    Args:
        coalesce_id: Node ID of the coalesce node
        raise_on_error: If True, raise GraphValidationError on incompatibility

    Returns:
        List of error messages (empty if compatible)

    Raises:
        GraphValidationError: If raise_on_error=True and schemas incompatible
    """
    errors = []

    # Get all incoming edges to coalesce
    incoming_edges = list(self._graph.in_edges(coalesce_id, data=True, keys=True))

    # Collect non-dynamic schemas
    incoming_schemas: list[tuple[str, type[PluginSchema]]] = []
    for from_node, _, _, _ in incoming_edges:
        from_info = self.get_node_info(from_node)
        schema = self._get_effective_producer_schema(from_node)

        # Check if schema is dynamic using SchemaConfig (source of truth)
        is_dynamic = (
            from_info.output_schema_config is not None
            and from_info.output_schema_config.is_dynamic
        )

        # Skip dynamic schemas in compatibility check
        if not is_dynamic and schema is not None:
            incoming_schemas.append((from_node, schema))

    # Compare pairwise
    if len(incoming_schemas) >= 2:
        base_node_id, base_schema = incoming_schemas[0]

        for other_node_id, other_schema in incoming_schemas[1:]:
            if not _schemas_compatible(base_schema, other_schema):
                error_msg = (
                    f"Coalesce node '{coalesce_id}' receives incompatible schemas. "
                    f"Branch from '{base_node_id}' has schema {base_schema.__name__}, "
                    f"but branch from '{other_node_id}' has schema {other_schema.__name__}. "
                    f"All branches merging at a coalesce must have compatible schemas."
                )
                if raise_on_error:
                    raise GraphValidationError(error_msg)
                errors.append(error_msg)

    return errors
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/core/test_dag.py::TestSchemaConfigPropagation::test_coalesce_validation_uses_schema_config -xvs
```

Expected: PASS

**Step 5: Run all DAG tests**

```bash
pytest tests/core/test_dag.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat: use SchemaConfig.is_dynamic in coalesce validation

- Replace _is_dynamic_schema() with SchemaConfig check
- Check from_info.output_schema_config.is_dynamic
- Eliminates second usage of Pydantic introspection
- Ref: P0-2026-01-24-eliminate-parallel-dynamic-schema-detection"
```

---

## Task 9: Remove _is_dynamic_schema Helper

**Goal:** Delete the introspection-based helper function (no longer needed).

**Files:**
- Modify: `src/elspeth/core/dag.py:67-84` (delete _is_dynamic_schema)
- Modify: `tests/core/test_dag.py:1948-1993` (delete old introspection tests)

**Step 1: Verify no remaining usages**

```bash
grep -n "_is_dynamic_schema" src/elspeth/core/dag.py
```

Expected: Only the function definition (lines 67-84), no other usages

**Step 2: Remove the helper function**

Delete lines 67-84 in `src/elspeth/core/dag.py`:

```python
# DELETE THIS ENTIRE FUNCTION:
def _is_dynamic_schema(schema: type[PluginSchema] | None) -> bool:
    """Check if a schema is dynamic (accepts any fields).

    Dynamic schemas have no defined fields and accept any extra fields.

    Args:
        schema: Schema class to check (None is treated as dynamic for backwards compat)

    Returns:
        True if schema is dynamic or None, False if explicit schema
    """
    if schema is None:
        return True  # Legacy: None = dynamic

    return (
        len(schema.model_fields) == 0  # No defined fields
        and schema.model_config.get("extra") == "allow"  # Accepts extra fields
    )
```

**Step 3: Remove old introspection tests**

Find and delete `TestDynamicSchemaDetection` class in `tests/core/test_dag.py` (around lines 1948-1993):

```bash
# Find the class
grep -n "class TestDynamicSchemaDetection" tests/core/test_dag.py
```

Delete the entire class (test_is_dynamic_schema_helper_detects_dynamic_schemas and related tests).

**Step 4: Run all DAG tests**

```bash
pytest tests/core/test_dag.py -v
```

Expected: All tests PASS (old introspection tests removed, new SchemaConfig tests pass)

**Step 5: Run integration tests**

```bash
pytest tests/integration/test_schema_validation_end_to_end.py -v
pytest tests/cli/test_plugin_errors.py::test_dynamic_schema_to_specific_schema_validation -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "refactor: remove _is_dynamic_schema introspection helper

- Delete Pydantic introspection function (no longer needed)
- Remove associated tests (replaced by SchemaConfig tests)
- Completes elimination of parallel detection mechanisms
- Ref: P0-2026-01-24-eliminate-parallel-dynamic-schema-detection"
```

---

## Task 10: Update no_bug_hiding.yaml Allowlist

**Goal:** Remove allowlist entry for `_is_dynamic_schema()` dict.get() usage.

**Files:**
- Modify: `config/cicd/no_bug_hiding.yaml`

**Step 1: Check current allowlist entry**

```bash
grep -A 5 "_is_dynamic_schema" config/cicd/no_bug_hiding.yaml
```

Expected: Entry exists for dag.py with _is_dynamic_schema usage

**Step 2: Remove the allowlist entry**

Find and remove the entry related to `_is_dynamic_schema` in `config/cicd/no_bug_hiding.yaml`.

If the entire dag.py entry was only for _is_dynamic_schema, remove the whole entry.
If there are other legitimate usages, just remove the _is_dynamic_schema line.

**Step 3: Verify no_bug_hiding check passes**

```bash
pytest tests/contracts/test_no_bug_hiding.py -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add config/cicd/no_bug_hiding.yaml
git commit -m "chore: remove _is_dynamic_schema from no_bug_hiding allowlist

- Remove allowlist entry for deleted introspection function
- Reduces allowlist surface area
- Ref: P0-2026-01-24-eliminate-parallel-dynamic-schema-detection"
```

---

## Task 11: Run Full Test Suite

**Goal:** Verify all tests pass with the new architecture.

**Step 1: Run complete test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All 3,279+ tests PASS

**Step 2: Run type checking**

```bash
mypy src/
```

Expected: Success: no issues found

**Step 3: Run linting**

```bash
ruff check src/
```

Expected: All checks passed

**Step 4: Verify no regressions in integration tests**

```bash
pytest tests/integration/ -v
pytest tests/cli/ -v
```

Expected: All tests PASS

---

## Task 12: Update Bug Ticket Status

**Goal:** Mark technical debt ticket as resolved.

**Files:**
- Modify: `docs/bugs/P0-2026-01-24-eliminate-parallel-dynamic-schema-detection.md`

**Step 1: Update ticket status**

At the end of `docs/bugs/P0-2026-01-24-eliminate-parallel-dynamic-schema-detection.md`, add:

```markdown
---

## Resolution

**Status:** ✅ **RESOLVED** (commit: <commit-hash>)

**Implementation:** Option A (SchemaConfig propagation) completed

**Changes Made:**
1. Extended `NodeInfo` with `input_schema_config` and `output_schema_config` fields
2. Updated `from_plugin_instances()` to extract SchemaConfig from all plugin types
3. Replaced `_is_dynamic_schema()` introspection with `SchemaConfig.is_dynamic` checks
4. Removed introspection helper and associated tests
5. Updated no_bug_hiding.yaml allowlist

**Verification:**
- ✅ All 3,279+ tests pass
- ✅ mypy type checking clean
- ✅ ruff linting clean
- ✅ Integration tests pass
- ✅ Only ONE detection mechanism remains (SchemaConfig.is_dynamic)
- ✅ No Pydantic introspection for dynamic detection
- ✅ Architecture quality improved from 2/5 to 4-5/5

**Implemented:** 2026-01-24
**Implementation Plan:** docs/plans/2026-01-24-eliminate-parallel-schema-detection.md
```

**Step 2: Get the final commit hash**

```bash
git log --oneline -1
```

Replace `<commit-hash>` in the ticket with the actual hash.

**Step 3: Commit ticket update**

```bash
git add docs/bugs/P0-2026-01-24-eliminate-parallel-dynamic-schema-detection.md
git commit -m "docs: mark P0-2026-01-24 tech debt as resolved

- Option A (SchemaConfig propagation) implemented
- Architecture quality improved from 2/5 to 4-5/5
- All verification criteria met"
```

---

## Completion Checklist

After completing all tasks:

- [ ] NodeInfo extended with SchemaConfig fields
- [ ] SchemaConfig extraction helper created
- [ ] Sources propagate SchemaConfig to graph
- [ ] Transforms propagate SchemaConfig to graph
- [ ] Sinks propagate SchemaConfig to graph
- [ ] Aggregations propagate SchemaConfig to graph
- [ ] Edge validation uses SchemaConfig.is_dynamic
- [ ] Coalesce validation uses SchemaConfig.is_dynamic
- [ ] _is_dynamic_schema() introspection helper deleted
- [ ] Old introspection tests deleted
- [ ] no_bug_hiding.yaml allowlist updated
- [ ] All tests pass (3,279+)
- [ ] mypy clean
- [ ] ruff clean
- [ ] Bug ticket marked resolved

**Success Metrics Achieved:**
- Detection mechanisms: 2 → 1 ✅
- Architecture quality: 2/5 → 4-5/5 ✅
- Pydantic coupling: HIGH → NONE ✅
- Maintenance locations: 2 → 1 ✅

---

## References

**Related Documentation:**
- Bug ticket: `docs/bugs/P0-2026-01-24-eliminate-parallel-dynamic-schema-detection.md`
- Parent bug: `docs/bugs/P0-2026-01-24-dynamic-schema-detection-regression.md`
- CLAUDE.md: "No Bug-Hiding Patterns" policy
- ARCHITECTURE.md: Trust Boundary Diagram

**Code Locations:**
- NodeInfo: `src/elspeth/core/dag.py:32-46`
- SchemaConfig: `src/elspeth/contracts/schema.py:166-249`
- Validation: `src/elspeth/core/dag.py:240-338`

**Test Coverage:**
- Unit tests: `tests/core/test_dag.py:TestSchemaConfigPropagation`
- Integration tests: `tests/integration/test_schema_validation_end_to_end.py`
- Contract tests: `tests/cli/test_plugin_errors.py`

---

## ⚠️ END OF OBSOLETE PLAN ⚠️

**This document is preserved for historical reference only.**

**Multi-agent review findings (2026-01-24):**
- Architecture review: Approved (4/5 result)
- Python review: Requested changes (CLAUDE.md violations)
- QA review: Requested changes (insufficient tests)
- Systems review: Approved with caveats (tactical win, strategic debt)

**Decision:** Instead of iterating on this plan to fix the violations and gaps, we chose to **fix the root cause** by moving validation to plugin construction time.

**Result:** Architecture improves to 5/5 (not 4/5), eliminates all parallel mechanisms, removes 200+ lines of validation code.

**Execute instead:** `docs/plans/2026-01-24-fix-schema-validation-properly.md`
