# ADR 003: Schema Validation Lifecycle

## Status

Accepted

## Context

Schema validation in ExecutionGraph was non-functional because the graph was built from config objects before plugins were instantiated. Schemas are instance attributes (`self.input_schema` set in plugin `__init__()`), so they weren't available during graph construction.

This was discovered through systematic debugging after investigating P2-2026-01-24-aggregation-nodes-lack-schema-validation. Root cause analysis revealed the issue affected ALL node types (sources, transforms, aggregations, gates, sinks).

## Decision

Restructure CLI to instantiate plugins BEFORE graph construction:

1. Load config (Pydantic models)
2. Instantiate ALL plugins (source, transforms, aggregations, sinks)
3. Build graph from plugin instances using `ExecutionGraph.from_plugin_instances()`
4. Extract schemas directly from instance attributes using `getattr()`
5. Validate graph (schemas populated, validation functional)
6. Execute pipeline using pre-instantiated plugins (no double instantiation)

## Consequences

### Positive

- **Schema validation now functional** - Detects incompatibilities at validation time
- **No double instantiation** - Plugins created once, reused in execution
- **Fail-fast principle** - Plugin instantiation errors occur during validation, not execution
- **Clean architecture** - Graph construction explicitly depends on plugin instances
- **No legacy code** - `from_config()` deleted immediately per CLAUDE.md
- **Coalesce support** - Fork/join patterns fully implemented

### Negative

- **Breaking change** - `from_config()` removed (but it never worked correctly anyway)
- **Plugin instantiation required for validation** - Can't validate without creating plugins
- **Resume command complexity** - Must override source with NullSource for resume operations

## Alternatives Considered

### Option A: Add schema fields to config models

Add `input_schema`/`output_schema` fields to Pydantic config models, populate via temporary plugin instantiation before graph construction.

**Rejected because:**
- Double instantiation (performance cost)
- Violates separation of concerns (config layer knows about plugin layer)
- Uses `object.__setattr__()` to bypass frozen models (hacky)
- Accumulates technical debt
- Violates CLAUDE.md no-legacy policy

### Option B: Extract schemas from plugin classes

Make schemas class attributes instead of instance attributes.

**Rejected because:**
- Many plugins compute schemas dynamically in `__init__()` based on config options
- Would require plugin API changes
- Doesn't support per-instance schema customization
- Example: CSVSource schema depends on `path` option

## Implementation

See implementation plans:
- `docs/plans/2026-01-24-schema-refactor-00-overview.md` - Overview and design
- `docs/plans/2026-01-24-schema-refactor-01-foundation.md` - Tasks 1-4
- `docs/plans/2026-01-24-schema-refactor-02-cli-refactor.md` - Tasks 5-7
- `docs/plans/2026-01-24-schema-refactor-03-testing.md` - Tasks 8-10
- `docs/plans/2026-01-24-schema-refactor-04-cleanup.md` - Tasks 11-15

## Key Technical Details

### PluginManager Changes

```python
# OLD (defensive - hides bugs)
def get_source_by_name(self, name: str) -> type[SourceProtocol] | None:
    for plugin in self.get_sources():
        if plugin.name == name:
            return plugin.plugin_class
    return None  # Caller must check for None

# NEW (explicit - crashes on bugs)
def get_source_by_name(self, name: str) -> type[SourceProtocol]:
    for plugin in self.get_sources():
        if plugin.name == name:
            return plugin.plugin_class
    available = [p.name for p in self.get_sources()]
    raise ValueError(f"Unknown source plugin: {name}. Available: {available}")
```

### Graph Construction

```python
# OLD (broken - schemas always None)
graph = ExecutionGraph.from_config(config, manager)
# Schemas extracted via getattr(config, "input_schema", None) → None

# NEW (working - schemas from instances)
plugins = instantiate_plugins_from_config(config)
graph = ExecutionGraph.from_plugin_instances(**plugins, gates=..., default_sink=...)
# Schemas extracted via getattr(instance, "input_schema", None) → actual schema
```

### Aggregation Dual-Schema

Aggregations have separate schemas for incoming and outgoing edges:
- `input_schema` - Individual rows entering aggregation
- `output_schema` - Batch results emitted after trigger

Validation handles this correctly:
```python
# Incoming edge to aggregation
consumer_schema = to_info.input_schema  # Individual row schema

# Outgoing edge from aggregation
producer_schema = from_info.output_schema  # Batch result schema
```

### Coalesce Implementation

Fork/join patterns fully supported:
```python
# Fork gate splits to multiple branches
gate_config.fork_to = ["branch_a", "branch_b"]

# Coalesce merges branches back
coalesce_config.branches = ["branch_a", "branch_b"]

# Graph construction creates:
# gate --branch_a--> coalesce
# gate --branch_b--> coalesce
# coalesce --continue--> sink
```

**Fork without coalesce:**
Branches not in a coalesce route directly to output sink:
```python
# Fork to separate destinations (no merge)
gate_config.fork_to = ["branch_a", "branch_b"]
# No coalesce defined

# Graph construction creates:
# gate --branch_a--> output_sink (fallback)
# gate --branch_b--> output_sink (fallback)
```

### Validation Semantics

**Dynamic Schema Behavior:**

Schema validation follows these rules:

1. **Dynamic schemas skip validation** - If either producer or consumer schema is `None` (dynamic), validation is skipped for that edge
2. **Mixed dynamic/specific pipelines are valid** - Dynamic schemas act as pass-through in validation
3. **Specific → Dynamic → Specific is valid** - Validation only checks specific → specific edges

**Examples:**

```python
# VALID: Dynamic source → Specific sink (validation skipped)
datasource:
  schema: dynamic
sinks:
  output:
    schema: {fields: {x: {type: int}}}

# VALID: Specific → Dynamic → Specific (dynamic in middle skipped)
datasource:
  schema: {fields: {x: {type: int}}}
transforms:
  - schema: dynamic  # Skipped in validation
sinks:
  output:
    schema: {fields: {x: {type: int}}}

# INVALID: Specific → Specific with incompatibility
datasource:
  schema: {fields: {x: {type: int}}}
sinks:
  output:
    schema: {fields: {y: {type: str}}}  # Missing field 'y'
```

**Gate Continue Routes:**

Gates support multiple routes resolving to "continue":
```python
gates:
  - name: filter
    routes:
      true: continue    # Routes to next gate or output
      false: rejected   # Routes to specific sink
```

**ALL routes resolving to "continue"** are handled, not just `"true"`.

**Fork/Join Validation:**

1. **Fork branches** inherit schema from upstream gate
2. **Coalesce merge** validates that all incoming branch schemas are compatible
3. **Fork without coalesce** validates each branch against its destination independently

**Fork Branch Explicit Destination Requirement:**

When a gate creates fork branches (via `fork_to` configuration), **every branch must have an explicit destination**. No fallback behavior is provided.

**Resolution order:**

1. **Explicit coalesce mapping** - If the branch name is listed in a coalesce's `branches` list → routes to that coalesce
2. **Explicit sink matching** - If the branch name exactly matches a sink name → routes to that sink
3. **Validation error** - If neither exists → graph construction crashes with `GraphValidationError`

**Valid Configuration:**
```python
gates:
  - name: categorize
    fork_to: [high_priority, low_priority]

coalesce:
  branches: [high_priority]  # high_priority joins coalesce

sinks:
  low_priority:  # Sink name matches branch name
    plugin: csv
    options: {path: low.csv}

# Result:
# - high_priority → coalesce (explicit)
# - low_priority → low_priority sink (explicit match)
```

**Invalid Configuration (will crash):**
```python
gates:
  - name: categorize
    fork_to: [high_priority, low_priority, medium_priority]

coalesce:
  branches: [high_priority]

sinks:
  low_priority: {plugin: csv}
  # medium_priority sink NOT defined

# Crashes with:
# "Gate 'categorize' has fork branch 'medium_priority' with no destination.
#  Fork branches must either:
#    1. Be listed in a coalesce 'branches' list, or
#    2. Match a sink name exactly
#  Available coalesce branches: ['high_priority']
#  Available sinks: ['low_priority']"
```

**Design Rationale:** Explicit-only destinations prevent silent configuration bugs:
- **Catches typos:** `categorry` (typo) instead of `category` crashes immediately
- **No hidden behavior:** Audit trail clearly shows intended routing
- **Fail-fast:** Missing destinations are caught at graph construction, not runtime
- **Aligns with CLAUDE.md:** No silent recovery, crash on configuration errors

**Alternative for implicit routing:** If you want fork branches to share a common destination, use gate `routes` instead:
```python
gates:
  - name: categorize
    condition: "row['priority'] in ['high', 'medium']"
    routes:
      true: prioritized_sink
      false: default_sink
```

This makes the routing explicit in the configuration.

**Critical invariants:**
- Every fork branch must have an explicit destination (coalesce or matching sink name)
- Fork branches without explicit destination crash during graph construction
- Coalesce validates incoming branch schema compatibility
- Gates with continue routes must have a next node in sequence

## Notes

**Fixes:**
- P0-2026-01-24-schema-validation-non-functional
- P2-2026-01-24-aggregation-nodes-lack-schema-validation
- P3-2026-01-24-coalesce-nodes-lack-schema-validation

**Timeline:** 4-5 days implementation + testing

**Review:** Multi-agent review (architecture-critic, python-code-reviewer, test-suite-reviewer, systems-thinking) validated approach and identified gaps

## Rollback Plan

If issues discovered post-deployment:

1. Revert commits from this ADR
2. Schema validation returns to non-functional state (acceptable short-term)
3. Investigate issues
4. Re-apply fix with corrections

**Note:** No backwards compatibility needed since `from_config()` never worked correctly.
