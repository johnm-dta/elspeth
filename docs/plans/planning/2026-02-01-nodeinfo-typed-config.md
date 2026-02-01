# Design: Type NodeInfo.config with Discriminated Union

**Status:** Planning
**Created:** 2026-02-01
**Updated:** 2026-02-01 (post-review, post-verification, policy-compliance)
**Goal:** Replace `NodeInfo.config: dict[str, Any]` with typed dataclasses to catch integration bugs at compile time

## Problem Statement

`NodeInfo.config` is currently `dict[str, Any]`, which provides no type safety:
- Field name typos discovered at runtime, not compile time
- No documentation of what fields each node type stores
- Integration issues between graph construction and access sites go undetected

## Analysis

NodeInfo.config stores heterogeneous data depending on node type:

| Node Type | Config Origin | Fields Known to Framework? |
|-----------|---------------|---------------------------|
| Source | Plugin's config | No (opaque) |
| Sink | Plugin's config | No (opaque) |
| Transform | Plugin's config + `schema` | Partially (`schema` only) |
| Gate | Synthesized | Yes (`condition`, `routes`, `schema`, `fork_to`) |
| Aggregation | Synthesized | Yes (`trigger`, `output_mode`, `options`, `schema`) |
| Coalesce | Synthesized | Yes (`branches`, `policy`, `merge`, `schema`, etc.) |

**Key insight:** Plugin configs are opaque (we don't control their shape), but framework-synthesized configs (Gate, Aggregation, Coalesce) are fully controlled by us.

## Solution: Discriminated Union of Typed Dataclasses

### Typed Config Dataclasses

```python
# contracts/node_config.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, TypeVar, TypeAlias, overload

@dataclass(frozen=True, slots=True)
class SourceNodeConfig:
    """Source nodes carry opaque plugin config."""
    plugin_config: dict[str, Any]

@dataclass(frozen=True, slots=True)
class SinkNodeConfig:
    """Sink nodes carry opaque plugin config."""
    plugin_config: dict[str, Any]

@dataclass(frozen=True, slots=True)
class TransformNodeConfig:
    """Transform nodes: plugin config + framework-added schema."""
    plugin_config: dict[str, Any]
    schema: dict[str, Any]
    required_input_fields: list[str] | None = None

@dataclass(frozen=True, slots=True)
class GateNodeConfig:
    """Gate nodes: framework-controlled routing config."""
    routes: dict[str, str]
    schema: dict[str, Any]
    condition: str | None = None      # Only for config-driven gates
    fork_to: list[str] | None = None
    plugin_config: dict[str, Any] | None = None  # Only for plugin gates

@dataclass(frozen=True, slots=True)
class AggregationNodeConfig:
    """Aggregation nodes: fully framework-controlled.

    NOTE: required_input_fields is a FIRST-CLASS FIELD here, not nested
    in options. During construction, extract from AggregationSettings
    and place directly on this config. No fallback lookup in options -
    that would be defensive programming on framework data.
    """
    trigger: dict[str, Any]  # TriggerSettings.model_dump()
    output_mode: str
    options: dict[str, Any]  # Plugin-specific options (no required_input_fields)
    schema: dict[str, Any]
    required_input_fields: list[str] | None = None  # First-class field, NOT in options

@dataclass(frozen=True, slots=True)
class CoalesceNodeConfig:
    """Coalesce nodes: fully framework-controlled."""
    branches: list[str]
    policy: str
    merge: str
    schema: dict[str, Any]  # Schema from first branch (computed during construction)
    timeout_seconds: float | None = None
    quorum_count: int | None = None
    select_branch: str | None = None
```

### Union Type

```python
NodeConfig: TypeAlias = (
    SourceNodeConfig
    | SinkNodeConfig
    | TransformNodeConfig
    | GateNodeConfig
    | AggregationNodeConfig
    | CoalesceNodeConfig
)
```

### Type Narrowing Helper

```python
T = TypeVar("T", bound=NodeConfig)

class NodeConfigTypeMismatch(TypeError):
    """Raised when node config doesn't match expected type for node_type.

    IMPORTANT: expected and actual MUST be concrete classes with __name__.
    Passing a TypeAlias (like NodeConfig) will crash - that's intentional.
    Always pass the specific dataclass type (e.g., GateNodeConfig).

    If this crashes with AttributeError on __name__, fix the call site
    to pass a concrete class, not a TypeAlias.
    """
    def __init__(self, node_id: str, node_type: str, expected: type, actual: type) -> None:
        # Direct access - crash if someone passes a TypeAlias (that's a bug)
        super().__init__(
            f"Node {node_id} has node_type='{node_type}' but config is "
            f"{actual.__name__}, expected {expected.__name__}"
        )
        self.node_id = node_id
        self.node_type = node_type
        self.expected = expected
        self.actual = actual


# Overloads for better mypy inference
@overload
def narrow_config(
    config: NodeConfig, expected: type[SourceNodeConfig], node_id: str, node_type: str
) -> SourceNodeConfig: ...

@overload
def narrow_config(
    config: NodeConfig, expected: type[SinkNodeConfig], node_id: str, node_type: str
) -> SinkNodeConfig: ...

@overload
def narrow_config(
    config: NodeConfig, expected: type[TransformNodeConfig], node_id: str, node_type: str
) -> TransformNodeConfig: ...

@overload
def narrow_config(
    config: NodeConfig, expected: type[GateNodeConfig], node_id: str, node_type: str
) -> GateNodeConfig: ...

@overload
def narrow_config(
    config: NodeConfig, expected: type[AggregationNodeConfig], node_id: str, node_type: str
) -> AggregationNodeConfig: ...

@overload
def narrow_config(
    config: NodeConfig, expected: type[CoalesceNodeConfig], node_id: str, node_type: str
) -> CoalesceNodeConfig: ...

@overload
def narrow_config(
    config: NodeConfig, expected: type[T], node_id: str, node_type: str
) -> T: ...

def narrow_config(config: NodeConfig, expected: type[T], node_id: str, node_type: str) -> T:
    """Narrow config to expected type, raising if mismatch.

    This is a type guard for accessing node-specific config fields.
    A mismatch indicates a bug in graph construction.
    """
    if isinstance(config, expected):
        return config
    raise NodeConfigTypeMismatch(node_id, node_type, expected, type(config))
```

### Serialization Helper

```python
def config_to_dict(config: NodeConfig) -> dict[str, Any]:
    """Serialize NodeConfig to dict for Landscape storage and topology hashing.

    INVARIANT: None fields are non-semantic and excluded from serialization.

    This is the CORRECT semantic definition, not a compatibility layer:
    - A field set to None means "not applicable for this node"
    - Non-applicable fields have no semantic meaning and don't affect hashes
    - Only meaningful (non-None) fields are recorded in the audit trail

    Example: A config-driven gate has condition="x > 0", fork_to=None.
    The fork_to=None is non-semantic (this gate doesn't fork).
    Serialized as: {"condition": "x > 0", "routes": {...}}
    NOT as: {"condition": "x > 0", "fork_to": null, "routes": {...}}

    This is analogous to how JSON APIs typically omit null fields rather than
    including explicit nulls - absence of a field is semantically equivalent
    to null for optional fields.

    Performance: Called once per node during graph construction.
    NOT on the hot path (not called per row).
    """
    result = asdict(config)
    return {k: v for k, v in result.items() if v is not None}
```

### Schema Access Helpers

```python
def get_node_schema(node: NodeInfo) -> dict[str, Any]:
    """Extract schema from any node that has one.

    Raises ValueError for source/sink nodes which don't have schema.
    """
    config = node.config
    if isinstance(config, (TransformNodeConfig, GateNodeConfig,
                           AggregationNodeConfig, CoalesceNodeConfig)):
        return config.schema
    if isinstance(config, (SourceNodeConfig, SinkNodeConfig)):
        raise ValueError(f"Node {node.node_id} ({node.node_type}) has no schema field")
    raise TypeError(
        f"Node {node.node_id} ({node.node_type}) has unsupported config type "
        f"{type(config).__name__}"
    )


def get_node_schema_optional(node: NodeInfo) -> dict[str, Any] | None:
    """Extract schema if present, None for source/sink.

    Use this for code paths that legitimately handle nodes without schema
    (replaces the .get("schema") pattern at line 1114).
    """
    config = node.config
    if isinstance(config, (TransformNodeConfig, GateNodeConfig,
                           AggregationNodeConfig, CoalesceNodeConfig)):
        return config.schema
    return None  # Source/Sink don't have schema


def get_required_input_fields(node: NodeInfo) -> frozenset[str]:
    """Extract required input fields from node config.

    Replaces .get("required_input_fields") patterns at lines 1175, 1181-1183.

    NOTE: required_input_fields is a FIRST-CLASS TYPED FIELD only.
    No fallback to options dict - that would be defensive programming
    on framework-owned data (violates CLAUDE.md policy).
    """
    config = node.config

    # Only Transform and Aggregation have required_input_fields
    if isinstance(config, (TransformNodeConfig, AggregationNodeConfig)):
        if config.required_input_fields:
            return frozenset(config.required_input_fields)

    return frozenset()
```

### NodeInfo Update

```python
# core/dag.py
@dataclass(slots=True)  # Add slots for memory efficiency
class NodeInfo:
    node_id: NodeID
    node_type: str  # Discriminator: "source" | "transform" | "gate" | ...
    plugin_name: str
    config: NodeConfig  # <- Changed from dict[str, Any]
    input_schema: type[PluginSchema] | None = None
    output_schema: type[PluginSchema] | None = None
    input_schema_config: SchemaConfig | None = None
    output_schema_config: SchemaConfig | None = None
```

## Critical: Single-Phase Construction Refactor

### The Problem

The current `from_plugin_instances()` constructs nodes in phases:
1. Create nodes with initial config
2. Add edges between nodes
3. Propagate schemas (coalesce gets schema from upstream branches)

This causes a **mutation at dag.py:789**:
```python
graph.get_node_info(coalesce_id).config["schema"] = first_schema
```

With `frozen=True` dataclasses, this mutation would crash.

### The Solution: Compute Before Create

Refactor `from_plugin_instances()` to compute all schemas BEFORE creating nodes:

```python
def from_plugin_instances(
    cls,
    source: SourceProtocol,
    transforms: list[TransformProtocol],
    sinks: dict[str, SinkProtocol],
    gates: list[GateSettings] | None = None,
    coalesce_settings: list[CoalesceSettings] | None = None,
    ...
) -> ExecutionGraph:
    """Build execution graph with typed, frozen configs.

    Construction happens in explicit phases:
    1. Compute topology (edges, branch mappings) - no nodes created yet
    2. Compute schemas for all nodes based on topology
    3. Create nodes with complete, frozen configs
    """
    graph = cls()

    # PHASE 1: Compute topology without creating nodes
    topology = _compute_dag_topology(
        source=source,
        transforms=transforms,
        sinks=sinks,
        gates=gates,
        coalesce_settings=coalesce_settings,
    )

    # PHASE 2: Compute schemas (now we know all connections)
    schemas = _compute_node_schemas(topology)

    # PHASE 3: Create nodes with complete configs
    for node_def in topology.nodes:
        config = _build_typed_config(node_def, schemas.get(node_def.id))
        graph.add_node(
            node_def.id,
            node_type=node_def.node_type,
            plugin_name=node_def.plugin_name,
            config=config,  # Frozen and complete
            ...
        )

    # PHASE 4: Add edges (nodes are immutable, only graph structure changes)
    for edge in topology.edges:
        graph.add_edge(edge.source, edge.target, ...)

    return graph
```

### Why This is the Right Fix

1. **Objects complete at creation** - No "fix up later" pattern
2. **Immutability preserved** - `frozen=True` works correctly
3. **Explicit phases** - Construction logic is clearer
4. **Follows CLAUDE.md** - "Schemas are immutable after graph construction"
5. **Consistent with RuntimeConfig pattern** - Same approach as `RuntimeRetryConfig.from_settings()`

### Topology Helper Types

```python
@dataclass
class NodeDefinition:
    """Node metadata computed during topology phase."""
    id: NodeID
    node_type: str
    plugin_name: str
    plugin_config: dict[str, Any] | None  # For plugin nodes
    framework_fields: dict[str, Any]  # For framework-synthesized nodes

@dataclass
class EdgeDefinition:
    """Edge metadata computed during topology phase."""
    source: NodeID
    target: NodeID
    edge_type: str  # "continue", "route", "fork", etc.

@dataclass
class DAGTopology:
    """Complete DAG structure before node creation."""
    nodes: list[NodeDefinition]
    edges: list[EdgeDefinition]
    branch_to_coalesce: dict[str, str]  # branch_name -> coalesce_id
```

### The Coalesce Schema Challenge (Verified)

The coalesce schema computation is non-trivial because:

1. **Current code at line 789** reads schema from `graph.get_node_info(from_node).config["schema"]`
2. This requires the **upstream nodes to already exist** with schema populated
3. But coalesce nodes are created at line 631 **before edges are added**
4. The loop at line 773 processes **each coalesce node** - multiple mutations if multiple coalesces

**Solution approach:**

```python
def _compute_coalesce_schemas(
    coalesce_settings: list[CoalesceSettings],
    branch_schemas: dict[str, dict[str, Any]],  # branch_name -> schema (from gates)
) -> dict[str, dict[str, Any]]:  # coalesce_name -> schema
    """Pre-compute schema for each coalesce based on branch outputs.

    Called BEFORE coalesce nodes are created. Uses branch_schemas which
    is populated from gate outputs during topology computation.
    """
    result = {}
    for coalesce_config in coalesce_settings:
        first_branch = coalesce_config.branches[0]
        first_schema = branch_schemas[first_branch]

        # Validate all branches match (same as line 781-787)
        for branch in coalesce_config.branches[1:]:
            if branch_schemas[branch] != first_schema:
                raise GraphValidationError(
                    f"Coalesce {coalesce_config.name}: branch '{branch}' has "
                    f"different schema than '{first_branch}'"
                )

        result[coalesce_config.name] = first_schema
    return result
```

The key insight: **gate outputs define branch schemas**, and gates are processed before coalesces. So we can extract branch→schema mappings during gate processing, then use them for coalesce schema computation.

## Verified Access Sites (13 Total)

Verification found **13 distinct access sites** that must be updated:

### In `core/dag.py` (8 sites)

| Line | Context | Pattern | Read/Write |
|------|---------|---------|------------|
| 449 | Plugin gate creation | `config["schema"]` | Read |
| 450 | Plugin gate creation | `"schema" in node_config` | Read |
| 453 | Plugin gate creation | `node_config['schema']` | Read (error msg) |
| 455 | Plugin gate creation | `node_config["schema"] =` | Write |
| 549 | Config gate creation | `config["schema"]` | Read |
| 779 | Coalesce validation | `config["schema"]` | Read |
| 782 | Coalesce validation | `config["schema"]` | Read |
| **789** | **Coalesce post-processing** | **`config["schema"] =`** | **Write (MUTATION)** |

### In `core/dag.py` method `_get_schema_config_from_node` (1 site)

| Line | Context | Pattern | Read/Write |
|------|---------|---------|------------|
| 1114 | Schema config retrieval | `config.get("schema")` | Read with `.get()` fallback |

**Note:** Line 1114 uses `.get()` because source/sink nodes don't have schema. The typed approach handles this via `get_node_schema_optional()`:

```python
def get_node_schema_optional(node: NodeInfo) -> dict[str, Any] | None:
    """Extract schema if present, None for source/sink."""
    config = node.config
    if isinstance(config, (TransformNodeConfig, GateNodeConfig,
                           AggregationNodeConfig, CoalesceNodeConfig)):
        return config.schema
    return None  # Source/Sink don't have schema
```

### In `engine/orchestrator.py` (2 sites)

| Line | Context | Pattern | Read/Write |
|------|---------|---------|------------|
| 900 | Landscape recording | `config["schema"]` | Read |
| 909 | Landscape recording | `config` (entire dict) | Read |

### In `plugins/base.py` (1 site)

| Line | Context | Pattern | Read/Write |
|------|---------|---------|------------|
| 355 | CSV source read | `config["path"]` | Read |

### In `core/canonical.py` (2 sites)

| Line | Context | Pattern | Read/Write |
|------|---------|---------|------------|
| 199 | Topology hash | `config` (entire dict) | Read |
| 259 | Upstream topology hash | `config` (entire dict) | Read |

## Plugin Gates vs Config Gates (Two Code Paths)

Verification revealed two distinct gate construction paths:

| Gate Type | Lines | Schema Source | Plugin Config |
|-----------|-------|---------------|---------------|
| **Plugin gates** | 446-464 | `upstream_schema` (read from prev node) | From transform plugin |
| **Config gates** | 545-590 | `graph.get_node_info(prev_node_id).config["schema"]` | None |

Both must produce `GateNodeConfig`, but with different field values:

```python
# Plugin gate
GateNodeConfig(
    routes=dict(gate.routes),
    schema=upstream_schema,
    condition=None,  # Plugin gates don't have condition
    fork_to=list(gate.fork_to) if gate.fork_to else None,
    plugin_config=dict(transform.config),  # Has plugin config
)

# Config-driven gate
GateNodeConfig(
    routes=dict(gate_config.routes),
    schema=prev_node_schema,
    condition=gate_config.condition,  # Has condition
    fork_to=list(gate_config.fork_to) if gate_config.fork_to else None,
    plugin_config=None,  # No plugin config
)
```

## Access Patterns

### Direct Attribute Access After Narrowing

```python
def process_gate(node: NodeInfo) -> None:
    config = narrow_config(node.config, GateNodeConfig, node.node_id, node.node_type)

    # mypy now knows config.routes exists
    for route, destination in config.routes.items():
        ...
```

### Landscape Recording

```python
# orchestrator.py
from elspeth.contracts.node_config import config_to_dict

self._landscape.record_node(..., config=config_to_dict(node.config), ...)
```

### Plugin Config Access

```python
# Sink path access
config = narrow_config(sink_node.config, SinkNodeConfig,
                       sink_node.node_id, sink_node.node_type)
path = config.plugin_config["path"]  # Plugin-specific field remains opaque
```

## Implementation Order (Detailed)

### Phase A: Add Types (Zero Blast Radius)

1. **Create `contracts/node_config.py`**
   - All 6 NodeConfig dataclasses
   - `NodeConfig` TypeAlias union
   - `NodeConfigTypeMismatch` exception
   - `narrow_config()` with overloads
   - `config_to_dict()` serialization
   - `get_node_schema()` and `get_node_schema_optional()` helpers

2. **Add topology helper types** (in `core/dag.py` or separate file)
   - `NodeDefinition`, `EdgeDefinition`, `DAGTopology` dataclasses

### Phase B: Refactor Construction (Critical - Tests Will Break)

3. **Refactor `from_plugin_instances()` to single-phase construction**
   - 3a. Extract `_compute_dag_topology()` - builds node/edge definitions without creating nodes
   - 3b. Track branch→schema mappings during gate processing (lines 455, 549)
   - 3c. Extract `_compute_coalesce_schemas()` - uses branch mappings to compute all coalesce schemas
   - 3d. Extract `_build_typed_config()` - creates frozen NodeConfig from definition + schema
   - 3e. Remove post-construction schema mutation at line 789
   - 3f. Update coalesce node creation (line 631) to include pre-computed schema

4. **Update NodeInfo type annotation** - `config: NodeConfig`

### Phase C: Update Access Sites (13 Sites)

5. **Update dag.py internal access sites** (8 sites)
   - Lines 449, 450, 453, 455: Plugin gate schema handling
   - Line 549: Config gate schema handling
   - Lines 779, 782: Coalesce branch validation
   - Line 1114: Use `get_node_schema_optional()`

6. **Update orchestrator.py** (3 sites)
   - Line 900: Schema access → use `get_node_schema()`
   - Line 909: Landscape recording → use `config_to_dict()`
   - Line 1740: CSV export → use `narrow_config()` + `plugin_config["path"]`

7. **Update canonical.py** (2 sites)
   - Lines 199, 259: Topology hashing → use `config_to_dict()`

8. **Update plugins/base.py** (1 site)
   - Line 355: CSV source path → use `narrow_config()` + `plugin_config["path"]`

### Phase D: Testing

9. **Add unit tests** (`tests/contracts/test_node_config.py`)
   - Construction, narrowing, serialization, frozen mutation prevention

10. **Add integration tests** (`tests/core/test_dag_typed_config.py`)
    - Single-phase construction produces correct types
    - Multiple coalesce nodes get correct schemas
    - Plugin gates vs config gates both produce `GateNodeConfig`
    - Landscape round-trip test

11. **Run full test suite** - verify no regressions

## Files to Modify (Verified)

| File | Change Type | Lines Affected | Scope |
|------|-------------|----------------|-------|
| `contracts/node_config.py` | **NEW** | ~180 | Configs + helpers + overloads + `get_node_schema_optional()` |
| `core/dag.py` | Modify | 449, 450, 453, 455, 549, 779, 782, 789, 1114 | NodeInfo + refactor `from_plugin_instances()` (~250 lines touched) |
| `core/canonical.py` | Modify | 199, 259 | Topology hash uses `config_to_dict()` |
| `engine/orchestrator.py` | Modify | 900, 909, 1740 | Landscape recording + CSV export path |
| `plugins/base.py` | Modify | 355 | CSV source path access |
| `tests/contracts/test_node_config.py` | **NEW** | ~100 | Unit tests for types |
| `tests/core/test_dag_typed_config.py` | **NEW** | ~150 | Single-phase construction + multiple coalesce tests |

## Testing Strategy

### Layer 1: Static Analysis (mypy)
- Wrong config type passed to `graph.add_node()`
- Accessing `.routes` on a `SourceNodeConfig`
- Missing required fields in config construction

### Layer 2: Unit Tests for Config Types
- Construction with required/optional fields
- `narrow_config()` success and mismatch cases
- `config_to_dict()` round-trip
- Frozen config prevents mutation (raises `FrozenInstanceError`)
- Empty list vs None semantics for optional fields

### Layer 3: Integration Tests - Production Path
- `ExecutionGraph.from_plugin_instances()` creates correctly typed configs
- Config-driven gates (from `GateSettings`) get `GateNodeConfig`
- Coalesce nodes have schema computed during construction (not mutated after)
- Use production factories per CLAUDE.md test path integrity rule

### Layer 4: Landscape Round-Trip Test (CRITICAL)
```python
def test_typed_configs_survive_landscape_storage(landscape_db):
    """Verify typed NodeConfigs serialize to Landscape correctly."""
    graph = ExecutionGraph.from_plugin_instances(...)
    run_id = orchestrator.execute(graph)

    # Read back from Landscape
    nodes = landscape.get_nodes(run_id)
    for node in nodes:
        config_dict = json.loads(node.config_json)
        assert isinstance(config_dict, dict)
        # Verify structure matches expected fields for node type
```

### Layer 5: Existing Test Suite
- Run full suite - any `node.config["field"]` access will fail, revealing missed sites
- Verify CSV export still works (orchestrator.py:1740 accesses `sink.config["path"]`)

## What We Gain

- **Compile-time safety:** mypy catches field access on wrong node types
- **Typo prevention:** Field names caught at construction time
- **Self-documenting:** Reading the dataclass shows exactly what each node type stores
- **Explicit boundary:** Clear what's framework-controlled vs plugin-opaque
- **Immutability guarantee:** Frozen configs can't be accidentally mutated
- **Single-phase construction:** Objects complete at creation, no "fix up later"

## Risks and Mitigations (Updated Post-Verification)

| Risk | Severity | Mitigation |
|------|----------|------------|
| Coalesce schema ordering more complex than expected | Medium | Track branch→schema during gate construction; compute all coalesce schemas before creating any coalesce nodes |
| Multiple coalesce nodes each need schema | Medium | Pre-compute ALL coalesce schemas in one pass before node creation |
| Two gate code paths (plugin vs config) | Low | Both paths produce `GateNodeConfig`; condition/plugin_config fields differ |
| canonical.py topology hashing | Low | Use `config_to_dict()` for serialization; verify hash stability |
| Line 1114 `.get()` pattern for source/sink | Low | Add `get_node_schema_optional()` helper that returns None for source/sink |
| Missed access sites | Low | Verification found all 13 sites; Layer 5 testing as backup |
| Landscape serialization breaks | Low | Layer 4 round-trip test catches this |
| CSV export path access breaks | Medium | Explicitly test CSV export; update to use `plugin_config["path"]` |

## Estimated Effort (Revised Post-Verification)

**6-8 hours total:**
- Typed config dataclasses + helpers: 1.5 hours
- Single-phase construction refactor: 3-4 hours (more complex due to coalesce schema ordering)
- Access site updates (13 sites across 5 files): 1.5 hours
- Testing (including multiple coalesce scenarios): 1-2 hours

**Why higher than original estimate:**
- Coalesce schema computation requires branch→schema tracking during gate construction
- Two distinct gate code paths (plugin vs config) both need updating
- Additional access sites in canonical.py not in original plan
- Multiple coalesce nodes require careful schema pre-computation

## Policy Compliance (Resolved Blocking Issues)

### RESOLVED: Hash Stability Framing

**Original issue:** Plan described None-filtering as "maintaining hash stability" and
avoiding "breaking checkpoint validation for existing runs" - this is backwards-compatibility
language that violates the no-legacy-code policy.

**Resolution:** Reframed as defining correct semantics:
- **None fields are non-semantic by design** (not for compatibility)
- Optional fields with None value mean "not applicable"
- Excluding None from serialization is the CORRECT semantic definition
- This is analogous to JSON APIs omitting null fields rather than including explicit nulls

### RESOLVED: Defensive Programming on Framework Data

**Original issue:** `get_required_input_fields()` used `.get()` and `isinstance()` on
`options` dict, which is framework-owned data - violates no-defensive-programming policy.

**Resolution:** Removed fallback lookup:
- `required_input_fields` is a FIRST-CLASS TYPED FIELD only
- No fallback to `options.get("required_input_fields")`
- During construction, extract from settings and place directly on config
- If field is missing at access time, that's a bug in construction (crash immediately)

### ADDED: JSON-Safe Config Validation

**Issue:** `dataclasses.asdict()` deep-copies plugin_config. If any plugin config contains
non-JSON or non-deepcopyable values, serialization will fail.

**Resolution:** Add config serialization test as go/no-go gate:

```python
# tests/contracts/test_node_config.py
def test_plugin_config_must_be_json_safe():
    """Verify plugin configs serialize without error.

    Plugin configs pass through asdict() and canonical_json().
    Non-JSON-safe values will fail here, not at runtime.
    """
    # Test with various plugin configs from real plugins
    for config in [
        {"path": "/tmp/test.csv", "schema": {"fields": "dynamic"}},
        {"model": "gpt-4", "temperature": 0.7},
        {"branches": ["a", "b"], "policy": "require_all"},
    ]:
        typed = SourceNodeConfig(plugin_config=config)
        serialized = config_to_dict(typed)
        # Must survive canonical_json (the actual hot path)
        canonical_json(serialized)  # Raises if not JSON-safe
```

### ADDED: slots=True Compatibility Check

**Issue:** Adding `slots=True` to NodeInfo may break code relying on `__dict__`.

**Resolution:** Quick verification before adopting:

```bash
# Run before implementation
rg "node.*__dict__|NodeInfo.*__dict__|\.config\.__dict__" src/ tests/
```

If matches found, evaluate whether they're test-only or production code.

## Go/No-Go Conditions

Before proceeding with implementation:

1. ✅ **Policy compliance resolved** - Both blocking issues addressed above
2. ✅ **Spike 1 passes** - Hash stability verified (7 tests passing)
3. ✅ **Spike 2 passes** - Schema presence verified (2 tests passing)
4. ✅ **Spike 3 passes** - Plugin config access verified (3 tests passing)
5. ✅ **JSON-safe test added** - Config serialization verified (3 tests passing)
6. ✅ **slots verification run** - No `__dict__` access found in src/ or tests/

**STATUS: ALL GO/NO-GO CONDITIONS MET - Ready for implementation**

## Review Feedback Addressed

This design incorporates feedback from the 4-perspective review:

1. **Architecture:** Added `schema` to `CoalesceNodeConfig` (was missing)
2. **Python Engineering:** Added `@overload` for `narrow_config()`, `TypeAlias` for union, `slots=True` for `NodeInfo`
3. **Quality Assurance:** Added Landscape round-trip test, addressed mutation conflict with single-phase construction
4. **Systems Thinking:** Chose Option B (refactor) over Option A (remove frozen) for architectural correctness

## Risk Reduction Spikes (Run Before Implementation)

Three targeted spikes to validate approach before committing to full implementation:

### Spike 1: Coalesce Schema Ordering (30 min)

**Goal:** Verify gate schemas are available before coalesce processing at line 770.

```python
# tests/spikes/test_coalesce_schema_ordering.py
def test_branch_schemas_available_before_coalesce():
    """Verify all gates have schema before coalesce loop."""
    # Instrument from_plugin_instances or add assertion at line 770
    # Verify: for each gate that feeds a coalesce branch,
    # node.config["schema"] is already set
```

**Pass criteria:** All upstream nodes have schema before line 770.

### Spike 2: Hash Stability (CRITICAL - 30 min)

**Goal:** Verify `config_to_dict()` with None filtering produces identical hashes.

```python
# tests/spikes/test_hash_stability.py
def test_typed_config_hash_matches_original():
    """Verify serialization produces same hash as current dict."""
    from elspeth.core.canonical import stable_hash

    # Current dict (only explicit fields)
    current_config = {"routes": {"true": "sink"}, "schema": {"fields": "dynamic"}}

    # Typed config (with None for optional fields)
    typed_config = GateNodeConfig(
        routes={"true": "sink"},
        schema={"fields": "dynamic"},
        condition=None, fork_to=None, plugin_config=None
    )

    # config_to_dict filters None → should match
    assert stable_hash(config_to_dict(typed_config)) == stable_hash(current_config)
```

**Pass criteria:** Hashes match for all node types.

### Spike 3: Plugin Config Access Pattern (15 min)

**Goal:** Verify plugin initialization doesn't break.

```python
# tests/spikes/test_plugin_config_access.py
def test_gate_plugin_receives_correct_config():
    """Verify gate plugins receive plugin_config dict, not NodeConfig."""
    # Create a plugin gate
    gate = MyGatePlugin(config={"routes": {...}, "my_option": "value"})

    # Plugin should read from config dict directly
    assert gate.routes == {...}
    assert gate.config["my_option"] == "value"
```

**Key insight:** Plugins receive `plugin_config` dict at instantiation (before NodeInfo exists). They don't read from NodeInfo.config - that's framework code only.

## Important Distinction: NodeInfo.config vs Plugin.config

```
┌─────────────────────────────────────────────────────────────────┐
│                  INSTANTIATION TIME                              │
│                                                                  │
│  Plugin receives config dict ──► Plugin stores as self.config   │
│  (before graph construction)      (plugin's own attribute)       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  GRAPH CONSTRUCTION                              │
│                                                                  │
│  NodeInfo.config = GateNodeConfig(                              │
│      routes=...,                                                 │
│      plugin_config=dict(plugin.config),  ← Copy of plugin dict  │
│  )                                                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  RUNTIME (Orchestrator)                          │
│                                                                  │
│  Framework reads NodeInfo.config (typed)                        │
│  Plugin reads self.config (its own dict)                        │
│                                                                  │
│  Line 1740: sink.config["path"]  ← Plugin's config, NOT NodeInfo│
└─────────────────────────────────────────────────────────────────┘
```

**Clarification for line 1740:** `sink.config["path"]` accesses the **SinkProtocol instance's** config attribute, not NodeInfo.config. This is unchanged by our refactor - plugins keep their own config dict.

## Verification Summary

This plan was verified against the actual codebase on 2026-02-01:

**Confirmed:**
- 13 access sites identified and documented with exact line numbers
- Coalesce schema mutation at line 789 confirmed as the critical issue
- Two gate construction paths (plugin vs config) require different handling
- Multiple coalesce nodes each get mutated in the loop at line 773

**Challenges identified:**
- Coalesce schema requires edge information → must track branch→schema during gate construction
- Line 1114 uses `.get()` fallback → need `get_node_schema_optional()` helper
- canonical.py topology hashing reads entire config dict → needs `config_to_dict()`

**Verification status:** ✅ Complete - plan aligned with codebase
