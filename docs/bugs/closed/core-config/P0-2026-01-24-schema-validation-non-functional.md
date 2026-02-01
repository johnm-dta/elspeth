## ✅ RESOLVED

**Status:** Fixed in RC-2
**Resolution:** Architectural refactor - plugin instantiation moved before graph construction
**Implementation:** See docs/plans/2026-01-24-schema-refactor-* (5 files)
**ADR:** See docs/design/adr/003-schema-validation-lifecycle.md

---

# Bug Report: Schema validation is non-functional (architectural issue)

## Summary

- Schema validation in `ExecutionGraph.validate()` is completely non-functional for ALL node types
- Root cause: Graph is built from config objects BEFORE plugins are instantiated
- Schemas are attached to plugin instances (`self.input_schema`) but graph construction uses config objects
- Result: `getattr(plugin_config, "input_schema", None)` always returns `None`, validation is silently skipped
- P2 aggregation bug is a symptom of this broader architectural flaw

## Severity

- Severity: **blocker** (core validation system is non-functional)
- Priority: **P0**

## Reporter

- Name or handle: systematic-debugging-session + architecture-critic + python-code-reviewer
- Date: 2026-01-24
- Related run/issue ID: Discovered during P2-2026-01-24-aggregation-nodes-lack-schema-validation investigation

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-4` @ `7c8a721`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: All environments affected
- Data set or fixture: All pipelines

## Steps To Reproduce

1. Create a pipeline with schema-incompatible transform chain
2. Configure transform T1 with `output_schema` producing `{field_a: str}`
3. Configure transform T2 with `input_schema` requiring `{field_b: int}`
4. Run `elspeth validate --settings pipeline.yaml`
5. Observe: Validation PASSES (should fail with schema incompatibility error)
6. Run `elspeth run --settings pipeline.yaml --execute`
7. Observe: Runtime failure when T2 tries to access missing `field_b`

## Expected Behavior

- `ExecutionGraph.validate()` should detect schema incompatibility
- Validation should fail with clear error: "Transform T1 -> T2: producer missing required fields {field_b}"
- Pipeline execution should not start

## Actual Behavior

- `ExecutionGraph.validate()` always passes (all schemas are `None`)
- Schema incompatibilities are not detected at validation time
- Runtime failures occur when plugins process incompatible data
- Silent validation bypass undermines audit trail integrity

## Evidence

### Code Flow Analysis (src/elspeth/cli.py)

```python
# Line 166: Load config (Pydantic models only)
config = load_settings(settings_path)

# Line 179: Build graph from config BEFORE plugins exist
graph = ExecutionGraph.from_config(config)
graph.validate()  # ❌ Schemas are all None here

# Lines 373-381: Plugins instantiated AFTER graph validation
for plugin_config in config.row_plugins:
    transform_cls = manager.get_transform_by_name(plugin_config.plugin)
    transforms.append(transform_cls(plugin_config.options))  # ✅ Schemas attached in __init__()
```

### Schema Extraction Pattern (src/elspeth/core/dag.py:455-456)

```python
# Transforms
graph.add_node(
    tid,
    node_type="transform",
    plugin_name=plugin_config.plugin,
    config=plugin_config.options,
    input_schema=getattr(plugin_config, "input_schema", None),  # ❌ Always None
    output_schema=getattr(plugin_config, "output_schema", None),  # ❌ Always None
)
```

**Why it's always None:**
- `plugin_config` is a `RowPluginSettings` Pydantic model (lines 392-406 in config.py)
- `RowPluginSettings` has fields: `plugin: str`, `options: dict[str, Any]`
- NO `input_schema` or `output_schema` fields exist on the config model
- `getattr(plugin_config, "input_schema", None)` therefore returns `None`

### Plugin Schema Definition (src/elspeth/plugins/transforms/batch_stats.py:84-98)

```python
class BatchStats(BaseTransform):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        # Schemas attached to plugin INSTANCE, not config object
        self.input_schema = create_schema_from_config(...)
        self.output_schema = create_schema_from_config(...)
```

### Validation Skip Logic (src/elspeth/core/dag.py:232-233)

```python
# Skip validation if either schema is None (dynamic)
if producer_schema is None or consumer_schema is None:
    continue  # ❌ ALWAYS skips because schemas are always None
```

## Impact

### User-facing impact
- Schema validation provides false confidence (passes when it shouldn't)
- Runtime failures occur instead of early validation failures
- Debugging is harder (error manifests during execution, not at config time)

### Data integrity / security impact
- **Critical:** Audit trail may record pipeline runs that processed incompatible data
- Schema mismatches can cause silent data corruption (wrong fields accessed, type errors swallowed)
- Undermines ELSPETH's auditability guarantee: "Every decision traceable to its source"

### Performance or cost impact
- Failed runs waste compute resources (should have been caught at validation)
- Manual debugging time for issues that should be automatically detected

## Root Cause Analysis

### Architectural Temporal Mismatch

**The Problem:**
1. `ExecutionGraph` is designed to validate schema compatibility
2. Schema validation requires access to `input_schema` and `output_schema`
3. Schemas are instance attributes on plugin objects (`self.input_schema = ...`)
4. Graph is built from `ElspethSettings` (config objects) before plugins exist
5. Config objects don't have schema fields
6. Result: Schemas are never available during graph construction/validation

**The Hidden Bug:**
- `getattr(plugin_config, "input_schema", None)` is a **symptom-hiding pattern**
- Returns `None` for missing attributes (defensive programming)
- Masks that the architecture fundamentally can't support schema validation
- Per CLAUDE.md "No Bug-Hiding Patterns" policy, this is exactly what we shouldn't do

**Why it wasn't caught:**
- Validation skip for `None` schemas is intentional (supports dynamic schemas)
- No integration tests verify schema validation works end-to-end
- The code "looks correct" - extraction logic exists, validation logic exists
- Bug is in the **data flow**, not the logic

## Proposed Fix

### Option A: Quick Fix (Minimal Changes)

**Add schema fields to config models, populate before graph construction**

**Step 1:** Add schema fields to Pydantic config models
```python
# src/elspeth/core/config.py
class DatasourceSettings(BaseModel):
    plugin: str
    options: dict[str, Any]
    output_schema: type[PluginSchema] | None = None  # NEW

class RowPluginSettings(BaseModel):
    plugin: str
    options: dict[str, Any]
    input_schema: type[PluginSchema] | None = None   # NEW
    output_schema: type[PluginSchema] | None = None  # NEW

class AggregationSettings(BaseModel):
    name: str
    plugin: str
    trigger: TriggerConfig
    output_mode: Literal["single", "passthrough", "transform"]
    options: dict[str, Any]
    input_schema: type[PluginSchema] | None = None   # NEW
    output_schema: type[PluginSchema] | None = None  # NEW

class SinkSettings(BaseModel):
    plugin: str
    options: dict[str, Any]
    input_schema: type[PluginSchema] | None = None   # NEW
```

**Step 2:** Populate schemas in CLI before graph construction
```python
# src/elspeth/cli.py (before line 179)
def _attach_schemas_to_config(config: ElspethSettings) -> ElspethSettings:
    """Instantiate plugins temporarily to extract schemas, attach to config."""
    manager = _get_plugin_manager()

    # Datasource
    source_cls = manager.get_source_by_name(config.datasource.plugin)
    source = source_cls(dict(config.datasource.options))
    config.datasource.output_schema = source.output_schema

    # Row plugins
    for plugin_config in config.row_plugins:
        transform_cls = manager.get_transform_by_name(plugin_config.plugin)
        transform = transform_cls(dict(plugin_config.options))
        plugin_config.input_schema = transform.input_schema
        plugin_config.output_schema = transform.output_schema

    # Aggregations
    for agg_config in config.aggregations:
        transform_cls = manager.get_transform_by_name(agg_config.plugin)
        transform = transform_cls(dict(agg_config.options))
        agg_config.input_schema = transform.input_schema
        agg_config.output_schema = transform.output_schema

    # Sinks
    for sink_name, sink_config in config.sinks.items():
        sink_cls = manager.get_sink_by_name(sink_config.plugin)
        sink = sink_cls(dict(sink_config.options))
        sink_config.input_schema = sink.input_schema

    return config

# In run() and validate() commands:
config = load_settings(settings_path)
config = _attach_schemas_to_config(config)  # NEW
graph = ExecutionGraph.from_config(config)
graph.validate()
```

**Step 3:** Update `from_config()` to extract aggregation schemas
```python
# src/elspeth/core/dag.py:482-487
graph.add_node(
    aid,
    node_type="aggregation",
    plugin_name=agg_config.plugin,
    config=agg_node_config,
    input_schema=getattr(agg_config, "input_schema", None),   # ADD
    output_schema=getattr(agg_config, "output_schema", None), # ADD
)
```

**Step 4:** Update `_validate_edge_schemas()` for aggregation dual-schema handling
```python
# src/elspeth/core/dag.py:208-246
def _validate_edge_schemas(self) -> list[str]:
    errors = []

    for edge in self.get_edges():
        from_info = self.get_node_info(edge.from_node)
        to_info = self.get_node_info(edge.to_node)

        # Get effective producer schema (handles gates)
        producer_schema = self._get_effective_producer_schema(edge.from_node)

        # Get consumer schema - handle aggregation dual schemas
        if to_info.node_type == "aggregation":
            # Incoming edge to aggregation: validate against input_schema
            consumer_schema = to_info.input_schema
        else:
            # Normal case: validate against input_schema
            consumer_schema = to_info.input_schema

        # Skip validation if either schema is None (dynamic)
        if producer_schema is None or consumer_schema is None:
            continue

        # ... rest of validation logic
```

**Pros:**
- Minimal code changes
- Preserves existing architecture
- Fixes validation for all node types

**Cons:**
- Instantiates plugins twice (once for schema extraction, once for execution)
- Config models become mutable (schemas attached dynamically)
- Violates separation of concerns (config layer knows about plugin layer)

---

### Option B: Correct Architecture (Larger Refactor)

**Restructure CLI to instantiate plugins before graph construction**

**Step 1:** Create `ExecutionGraph.from_plugin_instances()` class method
```python
# src/elspeth/core/dag.py
@classmethod
def from_plugin_instances(
    cls,
    source: SourceProtocol,
    transforms: list[TransformProtocol],
    sinks: dict[str, SinkProtocol],
    gates: list[GateSettings],
    aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]],
    output_sink: str,
) -> "ExecutionGraph":
    """Build graph from instantiated plugin instances.

    Schemas are extracted directly from plugin instances.
    """
    graph = cls()

    # Source node
    source_id = node_id("source", source.name)
    graph.add_node(
        source_id,
        node_type="source",
        plugin_name=source.name,
        config={},
        output_schema=source.output_schema,  # Extract from instance
    )

    # Transform nodes
    prev_node_id = source_id
    for i, transform in enumerate(transforms):
        tid = node_id("transform", transform.name)
        graph.add_node(
            tid,
            node_type="transform",
            plugin_name=transform.name,
            config={},
            input_schema=transform.input_schema,   # Extract from instance
            output_schema=transform.output_schema, # Extract from instance
        )
        graph.add_edge(prev_node_id, tid, label="continue", mode=RoutingMode.MOVE)
        prev_node_id = tid

    # ... similar for aggregations, gates, sinks

    return graph
```

**Step 2:** Refactor CLI to instantiate plugins first
```python
# src/elspeth/cli.py
def run(...):
    config = load_settings(settings_path)

    # Instantiate ALL plugins before building graph
    manager = _get_plugin_manager()

    source = manager.get_source_by_name(config.datasource.plugin)(
        dict(config.datasource.options)
    )

    transforms = [
        manager.get_transform_by_name(p.plugin)(dict(p.options))
        for p in config.row_plugins
    ]

    sinks = {
        name: manager.get_sink_by_name(s.plugin)(dict(s.options))
        for name, s in config.sinks.items()
    }

    # ... similar for aggregations

    # Build graph from instances (schemas available)
    graph = ExecutionGraph.from_plugin_instances(
        source=source,
        transforms=transforms,
        sinks=sinks,
        gates=list(config.gates),
        aggregations=aggregations,
        output_sink=config.output_sink,
    )

    # Validate with schemas attached
    graph.validate()

    # ... rest of execution
```

**Pros:**
- Clean separation of concerns
- Plugins instantiated once
- Schema extraction is straightforward (direct attribute access)
- Aligns with "fail fast" principle (plugins fail during instantiation if config invalid)

**Cons:**
- Larger refactor (new graph construction method)
- Changes CLI structure significantly
- May require updates to checkpoint/resume logic

---

## Recommendation

**Immediate (RC-1):** Option A - Quick fix
- Unblocks schema validation for RC-1 release
- Fixes P2 aggregation bug as part of broader fix
- Minimal risk to existing functionality

**Post-RC-1:** Option B - Architectural refactor
- Plan for next major version
- Create ADR documenting the decision
- Add to technical debt backlog

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (schema compatibility validation)
- Observed divergence: Validation completely non-functional
- Reason: Graph built from config before plugins exist
- Alignment plan: Implement Option A short-term, Option B long-term

## Acceptance Criteria

- [ ] Schema validation detects incompatible transform chains
- [ ] Schema validation detects incompatible source → transform edges
- [ ] Schema validation detects incompatible transform → sink edges
- [ ] Schema validation detects incompatible aggregation edges (dual-schema)
- [ ] Schema validation handles dynamic schemas (`None`) correctly
- [ ] Integration test verifies end-to-end schema validation works
- [ ] Documentation updated with schema lifecycle explanation

## Tests

### Existing tests to update:
- `tests/core/test_dag.py::TestSchemaValidation` - May need schema population
- `tests/core/test_dag.py::TestExecutionGraph` - Add schema validation tests

### New tests required:

**Test 1: Schema validation detects transform incompatibility**
```python
def test_schema_validation_catches_transform_incompatibility():
    """Verify schema validation fails on incompatible transform chain."""
    # Setup: T1 outputs {field_a: str}, T2 requires {field_b: int}
    # Expected: Validation error with missing field_b
```

**Test 2: Schema validation allows compatible chain**
```python
def test_schema_validation_allows_compatible_chain():
    """Verify schema validation passes on compatible transform chain."""
    # Setup: T1 outputs {field_a: str, field_b: int}, T2 requires {field_b: int}
    # Expected: Validation passes
```

**Test 3: Schema validation handles dynamic schemas**
```python
def test_schema_validation_skips_dynamic_schemas():
    """Verify schema validation skips nodes with None schemas."""
    # Setup: T1 has output_schema=None (dynamic)
    # Expected: Validation passes (skip)
```

**Test 4: Aggregation dual-schema validation**
```python
def test_aggregation_dual_schema_validation():
    """Verify aggregation incoming/outgoing edges validate separately."""
    # Setup: Source → Aggregation → Sink
    # Expected: Incoming edge validates against input_schema,
    #           Outgoing edge validates against output_schema
```

**Test 5: Integration test - end-to-end schema validation**
```python
def test_end_to_end_schema_validation_works():
    """Verify schema validation works in real pipeline construction."""
    # Setup: Load config, build graph, validate
    # Expected: Actual schemas attached, validation runs
```

## Notes / Links

- Related issues/PRs:
  - P2-2026-01-24-aggregation-nodes-lack-schema-validation (symptom)
  - P3-2026-01-24-coalesce-nodes-lack-schema-validation (symptom)
  - P1-2026-01-21-schema-validator-ignores-dag-routing (fixed gate inheritance)
  - Commits: `05fff54`, `234314b`, `f4dd59d` (gate schema fix)

- Related design docs:
  - `docs/contracts/plugin-protocol.md`
  - `CLAUDE.md` (Three-Tier Trust Model, No Bug-Hiding Patterns)

- Architecture review:
  - This bug reveals systematic-debugging + architecture-critic + python-code-reviewer findings

## Follow-up Work

1. **Create ADR:** Document schema lifecycle and validation architecture
2. **Add integration tests:** Verify schema validation works end-to-end
3. **Update P2 aggregation bug:** Mark as symptom of this broader issue
4. **Plan Option B refactor:** Post-RC-1 architectural cleanup
5. **Audit other `getattr(..., None)` patterns:** Check for similar symptom-hiding

## Implementation Complexity

**Option A (Quick Fix):**
- Estimate: **High** (4-6 hours)
- Requires changes in: config.py, dag.py, cli.py
- Test coverage: 5 new tests + updates to existing tests
- Risk: Medium (double instantiation has performance cost)

**Option B (Architectural Refactor):**
- Estimate: **Very High** (16-24 hours)
- Requires: New graph construction method, CLI restructure, checkpoint updates
- Test coverage: Extensive (all graph construction paths)
- Risk: High (large refactor near RC-1)

**Recommendation:** Option A for RC-1, plan Option B for next version
