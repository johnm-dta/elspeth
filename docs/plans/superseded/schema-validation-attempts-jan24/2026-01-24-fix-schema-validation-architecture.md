# Implementation Plan: Fix Schema Validation Architecture

**Bug:** P0-2026-01-24-schema-validation-non-functional
**Date:** 2026-01-24
**Status:** Proposed (awaiting approval)
**Target:** RC-1 (Option A) + Post-RC-1 (Option B)

---

## Executive Summary

Schema validation in `ExecutionGraph` is completely non-functional because the graph is built from config objects before plugins are instantiated. Schemas are attached to plugin instances in `__init__()`, but graph construction uses config objects that lack schema fields.

**Immediate Fix (Option A):** Add schema fields to config models, populate before graph construction
**Long-term Fix (Option B):** Restructure to instantiate plugins before building graph

---

## Option A: Quick Fix (Recommended for RC-1)

### Overview

Add schema fields to Pydantic config models and populate them by temporarily instantiating plugins before graph construction.

### Implementation Steps

#### Step 1: Add Schema Fields to Config Models

**File:** `src/elspeth/core/config.py`

**Changes:**

```python
# Line 380-390: DatasourceSettings
class DatasourceSettings(BaseModel):
    """Source plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name (csv_local, json, http_poll, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
    # NEW: Add schema field for validation
    output_schema: type[PluginSchema] | None = Field(
        default=None,
        description="Output schema from source plugin (populated during validation)",
    )


# Line 392-406: RowPluginSettings
class RowPluginSettings(BaseModel):
    """Transform plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
    # NEW: Add schema fields for validation
    input_schema: type[PluginSchema] | None = Field(
        default=None,
        description="Input schema for transform (populated during validation)",
    )
    output_schema: type[PluginSchema] | None = Field(
        default=None,
        description="Output schema from transform (populated during validation)",
    )


# Line 120-156: AggregationSettings
class AggregationSettings(BaseModel):
    """Aggregation configuration for batching rows."""

    model_config = {"frozen": True}

    name: str = Field(description="Aggregation identifier (unique within pipeline)")
    plugin: str = Field(description="Plugin name to instantiate")
    trigger: TriggerConfig = Field(description="When to flush the batch")
    output_mode: Literal["single", "passthrough", "transform"] = Field(
        default="single",
        description="How batch produces output rows",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
    # NEW: Add schema fields for dual-schema validation
    input_schema: type[PluginSchema] | None = Field(
        default=None,
        description="Input schema for individual rows (populated during validation)",
    )
    output_schema: type[PluginSchema] | None = Field(
        default=None,
        description="Output schema for aggregated result (populated during validation)",
    )


# Line 408-418: SinkSettings
class SinkSettings(BaseModel):
    """Sink plugin configuration per architecture."""

    model_config = {"frozen": True}

    plugin: str = Field(description="Plugin name (csv, json, database, webhook, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific configuration options",
    )
    # NEW: Add schema field for validation
    input_schema: type[PluginSchema] | None = Field(
        default=None,
        description="Input schema for sink (populated during validation)",
    )
```

**Note:** `model_config = {"frozen": True}` makes these models immutable after creation. We'll need to remove `frozen=True` or use a different approach (see "Frozen Model Challenge" below).

---

#### Step 2: Create Schema Attachment Function

**File:** `src/elspeth/cli.py` (new function before `run()` command)

**Add:**

```python
def _attach_schemas_to_config(config: ElspethSettings) -> ElspethSettings:
    """Temporarily instantiate plugins to extract schemas, attach to config objects.

    This function instantiates each plugin to access its input_schema and output_schema
    attributes, then attaches those to the config model for use during graph construction.

    Args:
        config: ElspethSettings with schema fields unset (None)

    Returns:
        ElspethSettings with schema fields populated from plugin instances

    Note:
        Plugins are instantiated twice: once here for schema extraction,
        once later for actual pipeline execution. This is a known limitation
        of Option A (quick fix). Option B (architectural refactor) eliminates
        double instantiation.
    """
    from elspeth.plugins import PluginManager

    manager = _get_plugin_manager()

    # Datasource schema
    source_cls = manager.get_source_by_name(config.datasource.plugin)
    if source_cls is None:
        available = [s.name for s in manager.get_sources()]
        raise ValueError(f"Unknown source plugin: {config.datasource.plugin}. Available: {available}")

    source = source_cls(dict(config.datasource.options))
    # Attach output_schema to datasource config
    # NOTE: Requires removing frozen=True or using object.__setattr__()
    object.__setattr__(config.datasource, "output_schema", source.output_schema)

    # Row plugin schemas
    for plugin_config in config.row_plugins:
        transform_cls = manager.get_transform_by_name(plugin_config.plugin)
        if transform_cls is None:
            available = [t.name for t in manager.get_transforms()]
            raise ValueError(f"Unknown transform plugin: {plugin_config.plugin}. Available: {available}")

        transform = transform_cls(dict(plugin_config.options))
        object.__setattr__(plugin_config, "input_schema", transform.input_schema)
        object.__setattr__(plugin_config, "output_schema", transform.output_schema)

    # Aggregation schemas (dual-schema handling)
    for agg_config in config.aggregations:
        transform_cls = manager.get_transform_by_name(agg_config.plugin)
        if transform_cls is None:
            available = [t.name for t in manager.get_transforms()]
            raise ValueError(f"Unknown aggregation plugin: {agg_config.plugin}. Available: {available}")

        transform = transform_cls(dict(agg_config.options))
        object.__setattr__(agg_config, "input_schema", transform.input_schema)
        object.__setattr__(agg_config, "output_schema", transform.output_schema)

    # Sink schemas
    for sink_name, sink_config in config.sinks.items():
        sink_cls = manager.get_sink_by_name(sink_config.plugin)
        if sink_cls is None:
            available = [s.name for s in manager.get_sinks()]
            raise ValueError(f"Unknown sink plugin: {sink_config.plugin}. Available: {available}")

        sink = sink_cls(dict(sink_config.options))
        object.__setattr__(sink_config, "input_schema", sink.input_schema)

    return config
```

---

#### Step 3: Call Schema Attachment Before Graph Construction

**File:** `src/elspeth/cli.py`

**Modify `run()` command (around line 179):**

```python
# Load and validate config via Pydantic
try:
    config = load_settings(settings_path)
except FileNotFoundError:
    typer.echo(f"Error: Settings file not found: {settings}", err=True)
    raise typer.Exit(1) from None
except ValidationError as e:
    typer.echo("Configuration errors:", err=True)
    for error in e.errors():
        loc = ".".join(str(x) for x in error["loc"])
        typer.echo(f"  - {loc}: {error['msg']}", err=True)
    raise typer.Exit(1) from None

# NEW: Attach schemas to config before graph construction
try:
    config = _attach_schemas_to_config(config)
except Exception as e:
    typer.echo(f"Error extracting schemas from plugins: {e}", err=True)
    raise typer.Exit(1) from None

# Build and validate execution graph (schemas now available)
try:
    graph = ExecutionGraph.from_config(config)
    graph.validate()
except GraphValidationError as e:
    typer.echo(f"Pipeline graph error: {e}", err=True)
    raise typer.Exit(1) from None
```

**Modify `validate()` command (around line 603):**

```python
# Load and validate config via Pydantic
try:
    config = load_settings(settings_path)
except FileNotFoundError:
    typer.echo(f"Error: Settings file not found: {settings}", err=True)
    raise typer.Exit(1) from None
except ValidationError as e:
    typer.echo("Configuration errors:", err=True)
    for error in e.errors():
        loc = ".".join(str(x) for x in error["loc"])
        typer.echo(f"  - {loc}: {error['msg']}", err=True)
    raise typer.Exit(1) from None

# NEW: Attach schemas to config before graph construction
try:
    config = _attach_schemas_to_config(config)
except Exception as e:
    typer.echo(f"Error extracting schemas from plugins: {e}", err=True)
    raise typer.Exit(1) from None

# Build and validate execution graph
try:
    graph = ExecutionGraph.from_config(config)
    graph.validate()
except GraphValidationError as e:
    typer.echo(f"Pipeline graph error: {e}", err=True)
    raise typer.Exit(1) from None

typer.echo("✅ Pipeline configuration valid!")
typer.echo(f"  Source: {config.datasource.plugin}")
typer.echo(f"  Transforms: {len(config.row_plugins)}")
typer.echo(f"  Aggregations: {len(config.aggregations)}")
typer.echo(f"  Sinks: {', '.join(config.sinks.keys())}")
typer.echo(f"  Graph: {graph.node_count} nodes, {graph.edge_count} edges")
```

---

#### Step 4: Update Graph Construction to Extract Aggregation Schemas

**File:** `src/elspeth/core/dag.py`

**Modify aggregation node construction (lines 482-487):**

```python
graph.add_node(
    aid,
    node_type="aggregation",
    plugin_name=agg_config.plugin,
    config=agg_node_config,
    input_schema=getattr(agg_config, "input_schema", None),   # NEW
    output_schema=getattr(agg_config, "output_schema", None), # NEW
)
```

---

#### Step 5: Update Schema Validation for Aggregation Dual-Schema Handling

**File:** `src/elspeth/core/dag.py`

**Modify `_validate_edge_schemas()` method (lines 208-246):**

```python
def _validate_edge_schemas(self) -> list[str]:
    """Validate schema compatibility along all edges.

    For each edge (producer -> consumer):
    - Get producer's effective output schema (walks through gates)
    - Get consumer's input schema
    - For aggregations: use input_schema for incoming edges, output_schema for outgoing edges
    - Check producer provides all required fields

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    for edge in self.get_edges():
        from_info = self.get_node_info(edge.from_node)
        to_info = self.get_node_info(edge.to_node)

        # Get effective producer schema (handles gates as pass-through)
        producer_schema = self._get_effective_producer_schema(edge.from_node)

        # Get consumer input schema - handle aggregation dual schemas
        if to_info.node_type == "aggregation":
            # Incoming edge to aggregation: validate against input_schema
            # Aggregations accept individual rows, then emit batch results
            consumer_schema = to_info.input_schema
        else:
            # Normal case: validate against input_schema
            consumer_schema = to_info.input_schema

        # Skip validation if either schema is None (dynamic)
        if producer_schema is None or consumer_schema is None:
            continue

        # Validate compatibility
        missing = _get_missing_required_fields(
            producer=producer_schema,
            consumer=consumer_schema,
        )

        if missing:
            errors.append(
                f"{from_info.plugin_name} -> {to_info.plugin_name} (route: {edge.label}): "
                f"producer missing required fields {missing}"
            )

    return errors
```

**Also update `_get_effective_producer_schema()` to handle aggregations (lines 248-302):**

```python
def _get_effective_producer_schema(self, node_id: str) -> type[PluginSchema] | None:
    """Get effective output schema for a node, walking through pass-through nodes.

    Gates and other pass-through nodes don't transform data - they inherit
    schema from their upstream producers. This method walks backwards through
    the graph to find the nearest schema-carrying producer.

    Aggregations DO transform data (input != output), so they use output_schema
    directly without inheritance.

    For gates with multiple incoming edges, all inputs must have compatible
    schemas (crashes if not - this is a graph construction bug).

    Args:
        node_id: Node to get effective schema for

    Returns:
        Output schema type, or None if node has no schema and no upstream producers

    Raises:
        GraphValidationError: If gate has no incoming edges or multiple inputs
            with incompatible schemas (graph construction bug)
    """
    node_info = self.get_node_info(node_id)

    # If node has output_schema, return it directly
    if node_info.output_schema is not None:
        return node_info.output_schema

    # Node has no schema - check if it's a pass-through type
    if node_info.node_type == "gate":
        # Gate passes data unchanged - inherit from upstream producer
        incoming = self.get_incoming_edges(node_id)

        if not incoming:
            # Gate with no inputs is a graph construction bug - CRASH
            raise GraphValidationError(
                f"Gate node '{node_id}' has no incoming edges - "
                f"this indicates a bug in graph construction"
            )

        # Get effective schema from first input
        first_schema = self._get_effective_producer_schema(incoming[0].from_node)

        # For multi-input gates, verify all inputs have same schema
        if len(incoming) > 1:
            for edge in incoming[1:]:
                other_schema = self._get_effective_producer_schema(edge.from_node)
                if first_schema != other_schema:
                    # Multi-input gates with incompatible schemas - CRASH
                    raise GraphValidationError(
                        f"Gate '{node_id}' receives incompatible schemas from "
                        f"multiple inputs - this is a graph construction bug. "
                        f"First input schema: {first_schema}, "
                        f"Other input schema: {other_schema}"
                    )

        return first_schema

    # NEW: Aggregations transform data - use output_schema directly (no inheritance)
    if node_info.node_type == "aggregation":
        # Aggregations have dual schemas: input_schema for incoming rows,
        # output_schema for batch results. When used as a producer,
        # return the output_schema (what they emit).
        # Note: If output_schema is None, we already returned it above.
        # This branch is unreachable but kept for clarity.
        return node_info.output_schema

    # Not a pass-through type and no schema - return None
    return None
```

---

### Frozen Model Challenge

**Problem:** Pydantic models with `model_config = {"frozen": True}` are immutable. We can't use normal assignment:

```python
config.datasource.output_schema = source.output_schema  # ❌ Raises FrozenInstanceError
```

**Solutions:**

**Option 1: Use `object.__setattr__()` (implemented above)**
```python
object.__setattr__(config.datasource, "output_schema", source.output_schema)  # ✅ Works
```

**Option 2: Remove `frozen=True` from config models**
```python
class DatasourceSettings(BaseModel):
    # model_config = {"frozen": True}  # REMOVE THIS LINE
    plugin: str
    options: dict[str, Any]
    output_schema: type[PluginSchema] | None = None
```

**Option 3: Create new instances with schemas (functional approach)**
```python
def _attach_schemas_to_config(config: ElspethSettings) -> ElspethSettings:
    # Create new datasource with schema attached
    new_datasource = config.datasource.model_copy(
        update={"output_schema": source.output_schema}
    )
    # Create new ElspethSettings with updated datasource
    return config.model_copy(update={"datasource": new_datasource})
```

**Recommendation:** Use Option 1 (`object.__setattr__()`) for minimal code changes. Document that schema attachment bypasses frozen protection intentionally.

---

### Testing Strategy

#### Unit Tests

**File:** `tests/core/test_dag.py`

**Add tests:**

```python
def test_schema_validation_detects_transform_incompatibility():
    """Verify schema validation fails on incompatible transform chain."""
    # Create schemas
    SchemaA = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"field_a": {"type": "str"}}}),
        "SchemaA",
    )
    SchemaB = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"field_b": {"type": "int"}}}),
        "SchemaB",
    )

    # Build graph with incompatible chain
    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=SchemaA)
    graph.add_node("t1", node_type="transform", plugin_name="mapper", input_schema=SchemaA, output_schema=SchemaA)
    graph.add_node("t2", node_type="transform", plugin_name="processor", input_schema=SchemaB, output_schema=SchemaB)
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=SchemaB)

    graph.add_edge("source", "t1", label="continue")
    graph.add_edge("t1", "t2", label="continue")  # INCOMPATIBLE: A → B
    graph.add_edge("t2", "sink", label="continue")

    # Validate
    errors = graph.validate()

    # Should detect missing field_b
    assert len(errors) > 0
    assert "field_b" in errors[0]


def test_aggregation_dual_schema_validation():
    """Verify aggregation incoming/outgoing edges validate separately."""
    # Input schema: individual rows with {value: float}
    InputSchema = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"value": {"type": "float"}}}),
        "InputSchema",
    )

    # Output schema: aggregated stats {count: int, sum: float}
    OutputSchema = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"count": {"type": "int"}, "sum": {"type": "float"}}}),
        "OutputSchema",
    )

    # Sink requires count and sum
    SinkSchema = create_schema_from_config(
        SchemaConfig.from_dict({"fields": {"count": {"type": "int"}, "sum": {"type": "float"}}}),
        "SinkSchema",
    )

    # Build graph: source → aggregation → sink
    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=InputSchema)
    graph.add_node(
        "agg",
        node_type="aggregation",
        plugin_name="batch_stats",
        input_schema=InputSchema,   # Accepts individual rows
        output_schema=OutputSchema,  # Emits aggregated stats
    )
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=SinkSchema)

    graph.add_edge("source", "agg", label="continue")
    graph.add_edge("agg", "sink", label="continue")

    # Validate - should pass
    errors = graph.validate()
    assert len(errors) == 0


def test_schema_validation_handles_dynamic_schemas():
    """Verify schema validation skips nodes with None schemas."""
    # Build graph with dynamic output schema
    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=None)  # Dynamic
    graph.add_node("t1", node_type="transform", plugin_name="mapper", input_schema=None, output_schema=None)
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=None)

    graph.add_edge("source", "t1", label="continue")
    graph.add_edge("t1", "sink", label="continue")

    # Validate - should pass (all schemas dynamic, validation skipped)
    errors = graph.validate()
    assert len(errors) == 0
```

#### Integration Tests

**File:** `tests/integration/test_schema_validation_end_to_end.py` (new file)

**Add tests:**

```python
def test_end_to_end_schema_validation_works():
    """Verify schema validation works in real pipeline construction from config."""
    import tempfile
    from pathlib import Path

    # Create test pipeline config with schema incompatibility
    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test_input.csv
    schema:
      fields:
        value: {type: float}

row_plugins:
  - plugin: field_mapper
    options:
      schema:
        fields:
          value: {type: float}
      mappings:
        value: price

sinks:
  output:
    plugin: csv
    options:
      path: test_output.csv
      schema:
        fields:
          total: {type: float}  # INCOMPATIBLE: requires 'total', gets 'price'

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        settings_path = Path(f.name)

    try:
        # Load config
        config = load_settings(settings_path)

        # Attach schemas
        config = _attach_schemas_to_config(config)

        # Build graph
        graph = ExecutionGraph.from_config(config)

        # Validate - should detect incompatibility
        errors = graph.validate()
        assert len(errors) > 0
        assert "total" in str(errors[0])

    finally:
        settings_path.unlink()
```

---

### Performance Considerations

**Double Instantiation:**
- Plugins instantiated twice: once for schema extraction, once for execution
- Performance impact: ~2x plugin `__init__()` calls
- Mitigation: Most plugins have lightweight `__init__()` (schema creation only)
- Future: Option B eliminates double instantiation

**Memory:**
- Temporary plugin instances created during schema extraction
- Garbage collected after `_attach_schemas_to_config()` returns
- Negligible impact for typical pipeline sizes

---

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Frozen models prevent schema assignment | Use `object.__setattr__()` to bypass |
| Double instantiation performance cost | Document as known limitation; plan Option B for next version |
| Plugin `__init__()` may fail during schema extraction | Wrap in try/except, provide clear error message |
| Existing pipelines may fail new validation | Document breaking change; provide migration guide |

---

### Rollout Plan

1. **PR 1:** Add schema fields to config models + frozen bypass pattern
2. **PR 2:** Implement `_attach_schemas_to_config()` function
3. **PR 3:** Update `from_config()` to extract aggregation schemas
4. **PR 4:** Update `_validate_edge_schemas()` for dual-schema handling
5. **PR 5:** Add unit tests + integration tests
6. **PR 6:** Update documentation (ADR, schema lifecycle)

---

## Option B: Architectural Refactor (Post-RC-1)

### Overview

Restructure CLI to instantiate plugins BEFORE building the graph. Graph construction extracts schemas directly from plugin instances.

### Implementation Sketch

**New method:** `ExecutionGraph.from_plugin_instances()`

**CLI changes:**
1. Instantiate all plugins first
2. Build graph from instances (schemas available)
3. Validate with schemas attached
4. Execute pipeline (no double instantiation)

**Benefits:**
- Clean architecture (single instantiation)
- Direct schema access (no frozen model workarounds)
- Aligns with "fail fast" principle

**Deferred to:** Post-RC-1 (larger refactor, higher risk)

---

## Acceptance Criteria

- [ ] Schema fields added to all config models
- [ ] `_attach_schemas_to_config()` implemented and tested
- [ ] Aggregation schemas extracted in `from_config()`
- [ ] Dual-schema validation works for aggregations
- [ ] Unit tests pass (5+ new tests)
- [ ] Integration test verifies end-to-end validation
- [ ] Documentation updated (ADR, schema lifecycle)
- [ ] No regressions in existing tests

---

## Timeline

**Option A (RC-1):**
- Day 1: Implement Steps 1-2 (config models + schema attachment)
- Day 2: Implement Steps 3-5 (CLI changes + DAG updates)
- Day 3: Unit tests + integration tests
- Day 4: Documentation + PR review
- **Total: 4 days**

**Option B (Post-RC-1):**
- Week 1: Design + ADR
- Week 2-3: Implementation + testing
- Week 4: Migration guide + rollout
- **Total: 4 weeks**

---

## References

- **Bug:** P0-2026-01-24-schema-validation-non-functional.md
- **Related:** P2-2026-01-24-aggregation-nodes-lack-schema-validation.md
- **Architecture:** ARCHITECTURE.md, CLAUDE.md (Three-Tier Trust Model)
- **Contracts:** docs/contracts/plugin-protocol.md
