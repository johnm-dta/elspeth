# ELSPETH Execution Graph Contract

> **Status:** FINAL (v1.1)
> **Last Updated:** 2026-02-08
> **Authority:** This document is the master reference for DAG construction, validation, and traversal.
> **Companions:** [Plugin Protocol](plugin-protocol.md) (Source/Transform/Sink), [System Operations](system-operations.md) (Gates/Forks/Coalesces).

## Overview

Every ELSPETH pipeline compiles to a **Directed Acyclic Graph (DAG)** before execution. The DAG is the single source of truth for how data flows — every edge, every routing decision, every schema contract is materialized in the graph at construction time, not discovered at runtime.

```
Pipeline YAML              Execution Graph              Row Processing
─────────────              ───────────────              ──────────────

source:                    ┌────────┐
  plugin: csv         ──►  │ source │
                           └───┬────┘
transforms:                    │ (continue)
  - plugin: enrich    ──►  ┌───┴────────┐
                           │ transform_0 │
gates:                     └───┬─────────┘
  - name: quality     ──►     │ (continue)
    routes:                ┌───┴──────┐
      true: continue       │ gate_0   │
      false: review        └─┬─────┬──┘
                             │     │
sinks:                  (continue)(review)
  output: ...         ──►    │     │
  review: ...         ──►  ┌─┴──┐ ┌┴──────┐
                           │sink│ │sink    │
                           │out │ │review  │
                           └────┘ └────────┘
```

### Why a DAG?

1. **Deterministic execution** — Topological sort guarantees consistent processing order across runs and resumes.
2. **Pre-flight validation** — Schema compatibility, reachability, and routing completeness are checked before any rows flow.
3. **Audit completeness** — Every routing path is an edge in the graph. If it is not in the graph, it cannot happen.
4. **Checkpoint/resume** — Deterministic node IDs (hash-based) ensure resumed runs reconnect to the same graph structure.

---

## Node Types

The graph contains six node types, each with specific roles and schema semantics.

### NodeInfo Contract

```python
@dataclass
class NodeInfo:
    node_id: NodeID                             # Deterministic hash-based ID
    node_type: str                              # "source", "transform", "gate", "aggregation", "coalesce", "sink"
    plugin_name: str                            # Plugin or operation name
    config: dict[str, Any]                      # Raw configuration
    input_schema: type[PluginSchema] | None     # What this node accepts
    output_schema: type[PluginSchema] | None    # What this node produces
    input_schema_config: SchemaConfig | None    # For contract validation
    output_schema_config: SchemaConfig | None   # For contract validation
```

### NodeType Enum

```python
class NodeType(StrEnum):
    SOURCE = "source"
    TRANSFORM = "transform"
    GATE = "gate"
    AGGREGATION = "aggregation"
    COALESCE = "coalesce"
    SINK = "sink"
```

### Node Semantics

| Type | Count | Input Schema | Output Schema | Modifies Row Data? | Stateful? |
|------|-------|-------------|---------------|-------------------|-----------|
| **Source** | Exactly 1 | N/A | Yes (declared) | N/A (produces rows) | No |
| **Transform** | 0+ | Yes | Yes | Yes | No (except aggregation) |
| **Gate** | 0+ | Yes | Pass-through | No | No |
| **Aggregation** | 0+ | Yes | Yes | Yes (batch processing) | Yes (buffer) |
| **Coalesce** | 0+ | Yes (multi-branch) | Pass-through (merged) | Yes (merge strategy) | Yes (pending tokens) |
| **Sink** | 1+ | Yes | N/A | N/A (consumes rows) | No |

**Pass-through semantics:** Gates and coalesces do not declare their own output schema. Instead, they inherit the schema from upstream. For coalesces, the effective schema is the intersection (union merge) or branch-specific (nested/select merge) of incoming branch schemas.

### Semantic Type Aliases

```python
NodeID = NewType("NodeID", str)
CoalesceName = NewType("CoalesceName", str)
BranchName = NewType("BranchName", str)
SinkName = NewType("SinkName", str)
GateName = NewType("GateName", str)
AggregationName = NewType("AggregationName", str)
```

These `NewType` aliases provide mypy-enforced type safety — you cannot accidentally pass a `SinkName` where a `NodeID` is expected.

---

## Edge Types

Edges represent the routing paths between nodes. The graph uses NetworkX `MultiDiGraph` to support multiple edges between the same node pair (e.g., a gate routing both "continue" and "review" to different sinks, or a fork creating multiple COPY edges from one gate).

### EdgeInfo Contract

```python
@dataclass(frozen=True)
class EdgeInfo:
    from_node: str
    to_node: str
    label: str           # Route label ("continue", sink name, branch name, error label)
    mode: RoutingMode    # MOVE, COPY, or DIVERT
```

### Edge Modes

| Mode | Semantics | Created By |
|------|-----------|------------|
| `MOVE` | Token exits current path, goes to destination only | `continue` edges, `route` edges |
| `COPY` | Token is cloned to destination (fork semantics) | Fork edges (gate → branch paths) |
| `DIVERT` | Token diverted to error/quarantine sink | Source `on_validation_failure`, transform `on_error` |

### Edge Label Conventions

| Edge Type | Label Format | Example |
|-----------|-------------|---------|
| Sequential flow | `"continue"` | source → transform_0 |
| Gate route | Sink name or `"continue"` | gate → high_values_sink |
| Fork branch | Branch name | gate → sentiment_path |
| Error divert | `"__error_{transform_seq}__"` | transform_2 → error_sink |
| Quarantine divert | `"__quarantine__"` | source → quarantine_sink |

> **Note:** Error and quarantine labels use the double-underscore ("dunder") convention to prevent collisions with user-defined sink names. These labels are generated by `error_edge_label()` in `contracts/enums.py`.

### Edge Uniqueness Constraint

From the same source node, no two outgoing edges may share the same label. This prevents routing ambiguity — if a gate has two edges labeled `"high_values"`, the engine cannot determine which to follow.

Enforced at: `ExecutionGraph.validate()`, unique edge labels check.

---

## Graph Construction

The graph is built by `ExecutionGraph.from_plugin_instances()`, which takes instantiated plugins and configuration and produces a validated DAG.

### Construction Signature

```python
@classmethod
def from_plugin_instances(
    cls,
    source: SourceProtocol,
    transforms: list[TransformProtocol],
    sinks: dict[str, SinkProtocol],
    aggregations: dict[str, tuple[TransformProtocol, AggregationSettings]],
    gates: list[GateSettings],
    coalesce_settings: list[CoalesceSettings] | None = None,
) -> ExecutionGraph
```

### Deterministic Node ID Generation

Node IDs are deterministic hashes of the node's configuration, ensuring that the same pipeline config always produces the same graph (critical for checkpoint/resume).

```python
def node_id(prefix: str, name: str, config: dict, sequence: int | None = None) -> NodeID:
    config_str = canonical_json(config)                              # RFC 8785 deterministic JSON
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]  # 48 bits of entropy

    if sequence is not None:
        return NodeID(f"{prefix}_{name}_{config_hash}_{sequence}")
    return NodeID(f"{prefix}_{name}_{config_hash}")
```

| Prefix | Node Type | Example |
|--------|-----------|---------|
| `source` | Source | `source_csv_a1b2c3d4e5f6` |
| `transform` | Transform (row) | `transform_enrich_f6e5d4c3b2a1_0` |
| `config_gate` | Config-driven gate | `config_gate_quality_check_1a2b3c4d5e6f` |
| `aggregation` | Aggregation | `aggregation_batch_stats_abcdef123456` |
| `coalesce` | Coalesce | `coalesce_merge_results_123abc456def` |
| `sink` | Sink | `sink_output_fedcba654321` |

The `sequence` suffix (for transforms only) prevents collisions when two transforms have identical configurations.

### Construction Phases

The graph is built in a strict order. Each phase depends on the results of previous phases.

```
Phase 1: Create Nodes
    │
    ├── Source node (single)
    ├── Sink nodes (one per named sink)
    ├── Transform chain (sequential)
    ├── Aggregation nodes (with trigger config)
    └── Coalesce nodes (with policy/strategy config)
    │
    ▼
Phase 2: Wire Edges
    │
    ├── Sequential "continue" edges (source → transform_0 → transform_1 → ...)
    ├── Config gate route edges (gate → sink, gate → continue)
    ├── Fork edges (gate → branch paths, COPY mode)
    ├── Coalesce incoming edges (branch paths → coalesce node)
    ├── Coalesce outgoing edges (coalesce → next node or default sink)
    └── DIVERT edges (error/quarantine routing)
    │
    ▼
Phase 3: Schema Validation (inside `from_plugin_instances()`)
    │
    ├── `validate_edge_compatibility()`: edge schema + contract checks
    └── `warn_divert_coalesce_interactions()`: non-fatal DIVERT warnings
    │
    ▼
Phase 4: Structural Validation (called externally by orchestrator)
    │
    └── `validate()`: acyclicity, reachability, source/sink counts, unique edge labels
```

#### Phase 1 Detail: Node Creation

**Source** (single):
- Extracts `output_schema` from source instance
- Stores `source_config` dict in node config

**Sinks** (one per named destination):
- Extracts `input_schema` from sink instance
- Maps `sink_name → node_id` in `_sink_id_map`

**Transforms** (sequential chain):
- Iterates through transforms in pipeline order
- Creates `continue` edges between sequential transforms
- Maps `transform_seq → node_id` in `_transform_id_map`

**Aggregations** (linked to transforms):
- Creates dual-schema nodes (input from upstream, output for downstream)
- Stores trigger config, output_mode, options
- Links via `continue` edge to the aggregation transform

**Config-driven gates** (after aggregations):
- Created from `GateSettings` config
- Expression validated at construction (not evaluation)
- Routes mapped in `_route_resolution_map`

**Coalesces** (before connecting gates):
- Validates branch uniqueness
- Maps branches → coalesce destinations in `_branch_to_coalesce`
- Computes insertion point (after latest producing gate)

#### Phase 2 Detail: Edge Wiring

**Fork connections:**
- Routes fork branches to explicit destinations only
- No fallback behavior — a branch name that doesn't match a coalesce or sink causes immediate validation error

**Gate continue routes:**
- `"continue"` label routes to the next node in the pipeline chain
- If the gate is the last gate, `"continue"` routes to the sink specified by the source or terminal transform's `on_success` option

**DIVERT edges** (structural error/quarantine routing):
- Source `on_validation_failure` → quarantine sink (if not `"discard"`)
- Transform `on_error` → error sink (if configured and not `"discard"`)
- Label format: `"__error_{transform_seq}__"` or `"__quarantine__"`
- Mode: `RoutingMode.DIVERT`

#### Phase 3 Detail: Validation

See "Graph Validation" section below.

### Internal ID Maps

The `ExecutionGraph` maintains several lookup maps for O(1) access during execution:

| Map | Type | Purpose |
|-----|------|---------|
| `_sink_id_map` | `dict[SinkName, NodeID]` | Sink name → node ID |
| `_transform_id_map` | `dict[int, NodeID]` | Transform sequence → node ID |
| `_config_gate_id_map` | `dict[GateName, NodeID]` | Config gate name → node ID |
| `_aggregation_id_map` | `dict[AggregationName, NodeID]` | Aggregation name → node ID |
| `_coalesce_id_map` | `dict[CoalesceName, NodeID]` | Coalesce name → node ID |
| `_branch_to_coalesce` | `dict[BranchName, CoalesceName]` | Fork branch → coalesce destination |
| `_route_label_map` | `dict[tuple[NodeID, str], str]` | (gate, sink_name) → route_label |
| `_route_resolution_map` | `dict[tuple[NodeID, str], str]` | (gate, label) → "continue" or sink_name |
| `_coalesce_gate_index` | `dict[CoalesceName, int]` | Coalesce name → producing gate pipeline index |

---

## Graph Validation

Validation is split across two methods, called at different times:

1. **`validate_edge_compatibility()`** — Called inside `from_plugin_instances()` after the graph is fully wired. Checks schema compatibility and contract satisfaction for every edge.
2. **`validate()`** — Called externally by the orchestrator before execution begins. Checks structural invariants (acyclicity, reachability, source/sink counts, unique edge labels).

Both must pass before any rows flow.

### Structural Validation (`validate()`)

| Check | Enforces | Error |
|-------|----------|-------|
| **Acyclicity** | No cycles in graph (it's a DAG) | `GraphValidationError` |
| **Single source** | Exactly one node with `node_type="source"` | `GraphValidationError` |
| **Sinks exist** | At least one sink node | `GraphValidationError` |
| **Reachability** | All nodes reachable from source (no orphans) | `GraphValidationError` |
| **Unique edge labels** | No duplicate outgoing labels from same node | `GraphValidationError` |

### Schema Compatibility Validation (`validate_edge_compatibility()`)

For every edge in the graph (except DIVERT edges), the producer's output schema must be compatible with the consumer's input schema.

```
Edge: transform_enrich → gate_quality_check
    │
    ├── Producer output: {id: int, name: str, amount: float, tier: str}
    ├── Consumer input:  {id: int, amount: float}
    │
    └── Compatible? YES — consumer requires subset of producer output
```

**Validation logic:**

1. **Skip DIVERT edges** — Quarantine/error sinks don't conform to producer schemas (they receive raw/error data).
2. **Resolve effective producer schema** — Walk backwards through pass-through nodes (gates, coalesce) to find the actual schema producer.
3. **Handle observed/dynamic schemas** — If either side is `OBSERVED` mode, bypass type validation (schema discovered at runtime).
4. **Check structural compatibility** — Consumer's required fields must exist in producer's output with compatible types.

**Pass-through schema resolution:**
- **Gates**: Inherit upstream schema (they don't declare their own output)
- **Coalesces**: Effective schema depends on merge strategy:
  - `union`: Intersection of all branch schemas
  - `nested`: New schema with branch names as keys
  - `select`: Selected branch's schema

### Contract Validation (Required/Guaranteed Fields)

Beyond type compatibility, the DAG validates that upstream nodes provide the fields downstream nodes require.

```yaml
# Source guarantees these fields
source:
  plugin: csv
  options:
    schema:
      guaranteed_fields: [customer_id, amount]

# Transform requires these fields
transforms:
  - plugin: llm_classifier
    options:
      required_input_fields: [customer_id, amount]
```

**Validation:**
- `guaranteed_fields` on producer ⊇ `required_input_fields` on consumer
- Missing fields → `ValueError` (fatal — pipeline will not start)

### DIVERT + Coalesce Interaction Warnings

After schema validation, `from_plugin_instances()` calls `warn_divert_coalesce_interactions()` to detect a dangerous pattern: if a transform between a fork gate and a `require_all` coalesce has `on_error` routing (a DIVERT edge), any row hitting that error path will cause the coalesce to permanently lose a branch.

These produce `GraphValidationWarning` objects (non-fatal) with code `DIVERT_COALESCE_REQUIRE_ALL`. The warnings are logged via structlog and returned to the caller.

```python
@dataclass(frozen=True)
class GraphValidationWarning:
    code: str                  # e.g., "DIVERT_COALESCE_REQUIRE_ALL"
    message: str               # Human-readable description
    node_ids: tuple[str, ...]  # Affected nodes (transform, gate, coalesce)
```

### Schema Modes

```python
@dataclass(frozen=True)
class SchemaConfig:
    mode: Literal["fixed", "flexible", "observed"]
    fields: tuple[FieldDefinition, ...] | None
    guaranteed_fields: tuple[str, ...] | None = None
    required_fields: tuple[str, ...] | None = None
    audit_fields: tuple[str, ...] | None = None       # Output fields NOT in the stability contract
```

| Mode | Fields | Extras Allowed | Type Validation |
|------|--------|----------------|-----------------|
| `fixed` | Declared upfront | No (extras rejected) | Full |
| `flexible` | Declared + extras OK | Yes | Declared fields only |
| `observed` | Discovered at runtime | Yes | None at construction time |

---

## Topological Ordering

The engine processes nodes in topological order — a linearization of the DAG where every node appears after all its predecessors.

```python
def topological_order(self) -> list[str]:
    try:
        return list(nx.topological_sort(self._graph))
    except nx.NetworkXUnfeasible as e:
        raise GraphValidationError(f"Cannot sort graph: {e}") from e
```

**Properties of topological order:**
- Source always first
- Sinks always last
- Transforms appear in pipeline sequence
- Parallel branches (fork paths) may interleave, but each path's internal order is preserved
- Coalesce appears after all its incoming branches

**Why NetworkX?** The DAG uses `networkx.MultiDiGraph` for graph operations. ELSPETH delegates cycle detection, topological sort, and reachability to NetworkX rather than reimplementing these algorithms — this is part of the acceleration stack philosophy (see CLAUDE.md).

---

## Processing Model

The `RowProcessor` traverses the DAG using a **work queue** model.

### Work Queue

```python
@dataclass(frozen=True, slots=True)
class TokenInfo:
    row_id: str                                  # Stable source row identity
    token_id: str                                # Instance of row in a specific DAG path
    row_data: PipelineRow                        # Row data (NOT dict — uses PipelineRow)
    branch_name: str | None = None               # Which fork path this token is on
    fork_group_id: str | None = None             # Groups children from a fork operation
    join_group_id: str | None = None             # Groups tokens merged in a coalesce
    expand_group_id: str | None = None           # Groups children from a deaggregation expand
```

Use `token.with_updated_data(new_data)` to create a new `TokenInfo` with updated `row_data` while preserving all lineage fields.

```python
@dataclass
class _WorkItem:
    token: TokenInfo
    start_step: int                      # Which step in transforms to start from
    coalesce_at_step: int | None = None  # Optional coalesce interception point
    coalesce_name: CoalesceName | None = None
```

### Traversal Algorithm

```
Source yields row
    │
    ▼
Create initial TokenInfo (row_id, token_id, row_data, ...)
    │
    ▼
Push _WorkItem(token, start_step=0) to work queue
    │
    ▼
While work queue is not empty:
    │
    ├── Pop next _WorkItem
    │
    ├── For step_index in range(start_step, len(transforms)):
    │   │
    │   ├── Is this step an aggregation node?
    │   │   YES → AggregationExecutor.accept()
    │   │         If held (buffering): break inner loop
    │   │         If flushed: handle output tokens
    │   │
    │   ├── Is this step a coalesce interception point?
    │   │   YES → CoalesceExecutor.accept()
    │   │         If held (waiting): break inner loop
    │   │         If merged: continue with merged token
    │   │
    │   ├── Is this step a config gate?
    │   │   YES → GateExecutor.execute_config_gate()
    │   │         │
    │   │         ├── CONTINUE → continue to next step
    │   │         ├── ROUTE → record sink_name, break inner loop
    │   │         └── FORK → push child _WorkItems, break inner loop
    │   │
    │   └── Is this step a transform?
    │       YES → TransformExecutor.execute_transform()
    │             Update token with result
    │
    ├── If sink_name is set:
    │   └── SinkExecutor.execute_sink(token, sink_name)
    │
    └── If no routing (fell through transforms):
        └── Route to sink specified by source or terminal transform's on_success option

Iteration guard: MAX_WORK_QUEUE_ITERATIONS = 10,000
```

### Fork Handling in the Work Queue

When a gate forks, the engine pushes one `_WorkItem` per child token:

```
Gate at step S returns FORK_TO_PATHS(["path_a", "path_b"])
    │
    ├── Create child tokens (T2 for path_a, T3 for path_b)
    │
    ├── For each child, look up coalesce destination and branch routing:
    │   If identity branch (no transforms) → start_node = coalesce_node (skip to merge)
    │   If transform branch → start_node = first transform in branch chain
    │   If no coalesce mapping → start_node = next pipeline node (continue normally)
    │
    ├── Push _WorkItem(T2, start_node=first_node_a, coalesce_node=coalesce_node, coalesce_name=...)
    └── Push _WorkItem(T3, start_node=first_node_b, coalesce_node=coalesce_node, coalesce_name=...)
```

> **Per-branch transforms (ARCH-15):** Fork branches can have intermediate transforms before reaching coalesce. When `CoalesceSettings.branches` uses dict format (`{branch_name: input_connection}`), the branch routes through the transform chain whose output matches the input connection. Identity branches (`branches: [a, b]`, normalized to `{a: a, b: b}`) skip directly to the coalesce node. See `ExecutionGraph.get_branch_first_nodes()` for the routing lookup.

### Coalesce Handling in the Work Queue

When a token reaches its `coalesce_at_step`:

```
_WorkItem has coalesce_at_step == current_step
    │
    ▼
CoalesceExecutor.accept(token, coalesce_name)
    │
    ├── held=True → break (token is waiting for siblings)
    │
    └── held=False, merged_token returned
        │
        ├── Push _WorkItem(merged_token, start_step=coalesce_step+1)
        └── Merged token continues through remaining pipeline
```

---

## Error and Quarantine Routing

DIVERT edges handle rows that fail validation or processing without crashing the pipeline.

### Source Validation Failures

```yaml
source:
  plugin: csv
  options:
    on_validation_failure: quarantine_sink   # Required
```

When a source row fails schema validation:
1. Source yields `SourceRow.quarantined(row, error, destination)`
2. Engine creates quarantine token
3. Token routed via DIVERT edge to `quarantine_sink`
4. `QuarantineEvent` recorded in audit trail
5. Pipeline continues with remaining rows

### Transform Processing Errors

```yaml
transforms:
  - plugin: price_calculator
    options:
      on_error: failed_calculations    # Optional
```

When a transform returns `TransformResult.error(...)`:
1. Engine checks `on_error` config for this transform
2. If configured: token routed via DIVERT edge to error sink
3. If `"discard"`: token dropped (audit event still recorded)
4. If not configured: `ConfigurationError` — pipeline crashes
5. `TransformErrorEvent` recorded in audit trail

### DIVERT Edge Properties

- DIVERT edges are **structural** — they exist in the graph at construction time
- They are **skipped** during schema compatibility validation (error/quarantine sinks receive non-conformant data by design)
- They use label format `"__error_{transform_seq}__"` or `"__quarantine__"`
- They always use `RoutingMode.DIVERT`

---

## Graph Access API

The `ExecutionGraph` class provides these methods for querying the graph:

### Node Queries

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_source()` | `NodeID \| None` | Get the single source node |
| `get_sinks()` | `list[NodeID]` | Get all sink nodes |
| `get_node_info(node_id)` | `NodeInfo` | Get metadata for any node |
| `get_nodes()` | `list[NodeInfo]` | Get all nodes |
| `has_node(node_id)` | `bool` | Check if node exists in graph |
| `node_count` | `int` | Number of nodes |

### Edge Queries

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_edges()` | `list[EdgeInfo]` | Get all edges as typed contracts |
| `get_incoming_edges(node_id)` | `list[EdgeInfo]` | Get incoming edges to node |
| `edge_count` | `int` | Number of edges |

### ID Map Accessors

All map accessors return copies to prevent external mutation of graph state.

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_sink_id_map()` | `dict[SinkName, NodeID]` | Sink name → node ID |
| `get_transform_id_map()` | `dict[int, NodeID]` | Transform sequence → node ID |
| `get_config_gate_id_map()` | `dict[GateName, NodeID]` | Config gate name → node ID |
| `get_aggregation_id_map()` | `dict[AggregationName, NodeID]` | Aggregation name → node ID |
| `get_coalesce_id_map()` | `dict[CoalesceName, NodeID]` | Coalesce name → node ID |
| `get_branch_to_coalesce_map()` | `dict[BranchName, CoalesceName]` | Fork branch → coalesce destination |
| `get_coalesce_gate_index()` | `dict[CoalesceName, int]` | Coalesce → producing gate pipeline index |

### Routing Queries

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_route_label(gate_id, sink_name)` | `str` | Edge label for gate→sink routing |
| `get_route_resolution_map()` | `dict` | All (gate, label) → destination mappings |

### Graph Operations

| Method | Returns | Purpose |
|--------|---------|---------|
| `topological_order()` | `list[str]` | Nodes in execution order |
| `is_acyclic()` | `bool` | DAG validity check |
| `validate()` | `None` | Structural validation (raises `GraphValidationError`) |
| `validate_edge_compatibility()` | `None` | Schema + contract validation (raises `ValueError`) |
| `warn_divert_coalesce_interactions(...)` | `list[GraphValidationWarning]` | Non-fatal DIVERT + coalesce interaction checks |
| `get_nx_graph()` | `MultiDiGraph` | Access underlying NetworkX graph |

---

## Linear Pipelines as Degenerate DAGs

A simple pipeline with no gates, forks, or coalesces compiles to a degenerate DAG — a single path from source to sink:

```
source → transform_0 → transform_1 → sink_output
```

All edges are `MOVE` mode with `"continue"` labels. This is the common case and benefits from the same validation, deterministic IDs, and audit trail as complex DAGs.

---

## Composite Primary Key: nodes Table

**CRITICAL:** The `nodes` audit table has a composite primary key `(node_id, run_id)`. The same `node_id` can exist in multiple runs when the same pipeline config runs multiple times (because node IDs are deterministic hashes of config).

**Queries touching `node_states` must use `node_states.run_id` directly:**

```python
# WRONG — Ambiguous join when node_id is reused across runs
query = (
    select(calls_table)
    .join(node_states_table, ...)
    .join(nodes_table, node_states_table.c.node_id == nodes_table.c.node_id)  # BUG!
    .where(nodes_table.c.run_id == run_id)
)

# CORRECT — Use denormalized run_id on node_states
query = (
    select(calls_table)
    .join(node_states_table, ...)
    .where(node_states_table.c.run_id == run_id)  # Direct filter
)
```

The `node_states` table has a denormalized `run_id` column (schema comment: "Added for composite FK"). Use it directly instead of joining through `nodes` table.

---

## Key Invariants

1. **Determinism** — Node IDs are deterministic hashes of config (RFC 8785 + SHA-256). Same config → same graph.
2. **Acyclicity** — The graph never contains cycles (enforced at validation).
3. **Reachability** — All nodes are reachable from the source (no orphan nodes).
4. **Single terminal state** — Each token reaches exactly one terminal outcome.
5. **Unique edge labels** — From the same node, no duplicate outgoing labels.
6. **Immutable schemas** — Schema contracts are frozen after first row (types locked).
7. **Atomic fork/coalesce** — Parent outcome recorded atomically with children.
8. **Audit completeness** — Every routing decision is traceable to an edge in the graph.
9. **No silent drops** — Every row that enters the pipeline reaches a terminal state.
10. **DIVERT edges are structural** — Error routing paths exist at construction time, not discovered at runtime.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-08 | Initial contract — Node types, edge types, graph construction, validation, schema propagation, processing model, error routing |
| 1.1 | 2026-02-08 | Accuracy pass — Fixed edge label formats (dunder convention), node ID prefixes, validation phase separation, fork semantics, contract error severity. Added TokenInfo contract, ID map accessors, GraphValidationWarning, audit_fields. |
