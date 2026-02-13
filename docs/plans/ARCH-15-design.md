# ARCH-15: Per-Branch Transforms Between Fork and Coalesce

**Date:** 2026-02-13
**Status:** REVIEWED — ready for implementation
**Tracked:** `elspeth-rapid-jyvr` (P2 feature)
**Branch:** `RC3-quality-sprint`
**Review:** 4 SME agents (architecture critic, systems thinker, Python reviewer, test analyst)

---

## Problem Statement

The DAG builder wires fork branches directly to coalesce nodes. No intermediate
transforms can be placed on individual branches. This means fork/coalesce is a
**merge barrier only** — it duplicates rows and reunites them, but cannot run
different processing on each branch.

The canonical use case this blocks:

```yaml
# DESIRED but currently impossible:
# Fork → sentiment API on branch A, entity API on branch B → merge enriched results
```

### Root Cause: Three Architectural Barriers

**Barrier 1 — Builder (connection system bypass):**
Fork branch names never enter the producer/consumer registry (`builder.py:333-345`).
Branches create direct COPY edges from gate → coalesce/sink. A transform declaring
`input: path_a` would fail validation because `path_a` has no registered producer.

**Barrier 2 — Runtime (coalesce jump):**
`DAGNavigator.create_continuation_work_item()` (`dag_navigator.py:240-263`) sends
fork children directly to the coalesce node when `coalesce_name` is set. Tokens
never visit intermediate nodes.

**Barrier 3 — Topological map (`node_to_next`):**
Built via `graph.get_next_node()` which only follows MOVE-mode "continue" edges.
Fork COPY edges are invisible to this traversal, so branch transform chains would
have no `node_to_next` entries even if they existed in the graph.

### What Does NOT Need Changing

Research confirmed these subsystems are already compatible:

| Subsystem | Why No Changes Needed |
|-----------|----------------------|
| **CoalesceExecutor** | Uses `token.branch_name` for identity, not routing topology |
| **TokenInfo** | Frozen dataclass; `with_updated_data()` preserves `branch_name` through transforms |
| **Merge policies** | `require_all`/`quorum`/`best_effort`/`first` all key on `branch_name` |
| **Merge strategies** | `union`/`nested`/`select` operate on arrived token data, topology-agnostic |
| **Checkpoint recovery** | Child tokens get individual outcomes; stateless transforms need no checkpoint state |
| **Guaranteed fields** | `get_effective_guaranteed_fields()` already returns INTERSECTION across branches |
| **Branch loss notification** | `notify_branch_lost()` uses `branch_name`, not graph position |

---

## Design

### User-Facing Configuration

**Current syntax (preserved unchanged):**

```yaml
gates:
- name: fork_gate
  input: preprocessed
  condition: "True"
  routes: { 'true': fork }
  fork_to: [path_a, path_b]

coalesce:
- name: merge_results
  branches: [path_a, path_b]       # list format — direct wiring (no transforms)
  policy: require_all
  merge: nested
```

**New syntax (per-branch transforms):**

```yaml
gates:
- name: fork_gate
  input: preprocessed
  condition: "True"
  routes: { 'true': fork }
  fork_to: [path_a, path_b]

transforms:
- name: sentiment_analysis
  plugin: openrouter
  input: path_a                     # consumes from fork branch
  on_success: sentiment_done
  options:
    prompt: "Classify sentiment: {text}"

- name: entity_extraction
  plugin: openrouter
  input: path_b                     # consumes from fork branch
  on_success: entities_done
  options:
    prompt: "Extract entities: {text}"

coalesce:
- name: merge_results
  branches:                         # dict format — maps branch identity → input connection
    path_a: sentiment_done          # "collect tokens arriving on 'sentiment_done' as branch path_a"
    path_b: entities_done           # "collect tokens arriving on 'entities_done' as branch path_b"
  policy: require_all
  merge: nested
```

**Key design decisions:**

1. **`branches` accepts both list and dict.** List = identity mapping (current behavior).
   Dict = explicit mapping of branch identity → input connection name. This is config
   ergonomics, not backwards compatibility — `branches: [a, b]` is cleaner than
   `branches: {a: a, b: b}` when no transforms are needed.
2. **Branch names become first-class connections** produced by the fork gate — but ONLY
   when a transform actually consumes from them. Identity-mapped branches keep direct
   COPY edges (no connection registration needed).
3. **Transform chains work naturally.** Multiple chained transforms on a branch use the
   existing `input`/`on_success` connection mechanism:
   ```yaml
   - name: enrich
     input: path_a
     on_success: enriched_a
   - name: classify
     input: enriched_a
     on_success: classified_a
   # coalesce branches: { path_a: classified_a }
   ```
4. **Branches without transforms** use identity mapping. In the dict format, write
   `path_b: path_b` (or use list format if NO branches have transforms).
5. **No new config sections or attributes.** Per-branch transforms are normal transforms
   that happen to consume from a fork branch connection. The only config change is that
   `branches` on `CoalesceSettings` can be a dict.
6. **No `RuntimeCoalesceConfig`.** `CoalesceSettings` is in `EXEMPT_SETTINGS` — it is
   consumed at DAG construction time, not runtime. No runtime config class, protocol,
   `from_settings()`, or alignment mapping needed. The only runtime artifact is the
   `_branch_first_node` topology map on `DAGNavigator`, derived from graph structure.

### Schema Validation Changes

**Two independent validation paths must both be relaxed:**

1. **Builder pass** (`builder.py:701-720`): Raw dict equality check on branch schemas.
2. **Graph pass** (`graph.py:972-1002`): `validate_edge_compatibility` structural check.

Both currently enforce identical schemas across ALL branches.

**New rule by merge strategy (applies to BOTH paths):**

| Strategy | Schema Requirement | Rationale |
|----------|-------------------|-----------|
| `union` | Compatible types on shared fields | Flat merge; type conflicts would corrupt data |
| `nested` | No constraint (any schema per branch) | Each branch keyed separately; no field collision possible |
| `select` | No constraint (only selected branch matters) | Non-selected branches are discarded |

**Schema population for coalesce nodes** must be strategy-aware (not dependent on
NetworkX iteration order of incoming edges):
- `union`: Merged schema from all branches
- `nested`: New schema with branch names as object fields
- `select`: Selected branch's schema

---

## Implementation Plan

### Layer 1: Config Schema (`core/config.py`)

**Change `CoalesceSettings.branches`** to `dict[str, str]` with a before-validator
that normalizes list input:

```python
branches: dict[str, str] = Field(..., min_length=2)

@field_validator("branches", mode="before")
@classmethod
def normalize_branches(cls, v: Any) -> dict[str, str]:
    """Normalize list format to dict (identity mapping)."""
    if isinstance(v, list):
        return {b: b for b in v}
    return v
```

This approach (per Python review):
- Declares the canonical type honestly (`dict[str, str]`) — no type lies to mypy
- Uses `field_validator(mode="before")` so all subsequent validators see dict consistently
- Avoids `object.__setattr__` on frozen model
- Avoids validator ordering conflicts with existing `validate_branch_names`

**Update `validate_branch_names`** to validate both keys AND values:
- Keys: valid branch names (existing rules)
- Values: valid connection/sink names

**No `RuntimeCoalesceConfig` changes.** CoalesceSettings is in `EXEMPT_SETTINGS` —
consumed at DAG construction time, not runtime. No protocol, alignment, or
`from_settings()` changes needed.

### Layer 2: DAG Builder (`core/dag/builder.py`)

**Step 2a — Conditionally register fork branches as produced connections:**

Only register a branch name as a produced connection when a transform actually
consumes from it (i.e., when `branch_input != branch_name` in the coalesce config).
This avoids creating dangling producers that fail the `_validate_connection_namespaces`
check.

```python
# Determine which branches have transforms (input != identity)
transformed_branches: set[str] = set()
for coalesce_config in coalesce_settings:
    for branch_name, input_connection in coalesce_config.branches.items():
        if input_connection != branch_name:
            transformed_branches.add(branch_name)

# Only register as producer if a transform will consume from it
for branch_name in gate_entry.fork_to:
    if branch_name in transformed_branches:
        register_producer(
            connection_name=branch_name,
            node_id=gate_entry.node_id,
            label=branch_name,
            description=f"Fork branch '{branch_name}' from gate '{gate_entry.name}'"
        )
```

**Step 2b — Branch-to-coalesce edge creation (split by transform presence):**

```python
for branch_name, input_connection in coalesce_config.branches.items():
    if input_connection == branch_name:
        # Identity mapping — direct COPY edge (current behavior)
        graph.add_edge(gate_node_id, coalesce_id, label=branch_name, mode=RoutingMode.COPY)
    else:
        # Transform chain — register coalesce as consumer of the final connection
        register_consumer(
            connection_name=input_connection,
            node_id=coalesce_id,
            description=f"Coalesce '{coalesce_config.name}' branch '{branch_name}'"
        )
        # The connection resolution system (producer/consumer matching) will create
        # the MOVE edges through the transform chain automatically
```

**Step 2c — Strategy-aware schema validation (both paths):**

Replace strict identical-schema checks in BOTH `builder.py:701-720` AND
`graph.py:972-1002` with strategy-aware validation:

```python
if coalesce_config.merge == "union":
    _validate_union_schema_compatibility(branch_schemas)
elif coalesce_config.merge in ("nested", "select"):
    pass  # No cross-branch schema constraint
```

For schema population on the coalesce node, use strategy-aware logic instead of
relying on NetworkX iteration order of incoming edges.

**Step 2d — Update `branch_to_coalesce` building (use dict keys):**

```python
for branch_name in coalesce_config.branches.keys():
    branch_to_coalesce[BranchName(branch_name)] = CoalesceName(coalesce_config.name)
```

**Step 2e — Add branch transform chain nodes to graph metadata:**

Branch transform nodes must be added to:
- `node_to_plugin` map (so the processor can execute them)
- `node_step_map` (for audit trail step indexing)
- `node_to_next` map (via MOVE "continue" edges, automatic from `get_next_node()`)

### Layer 3: Runtime Routing (`engine/dag_navigator.py`, `engine/processor.py`)

**Step 3a — `_branch_first_node` mapping (ALL branches populated):**

Populate for ALL branches — identity branches map to the coalesce node, transformed
branches map to the first transform's node ID. This eliminates the need for
defensive `.get()` (which violates the project's defensive programming prohibition).

```python
# Built during DAGNavigator initialization
# Uses MappingProxyType for consistency with other navigator maps
self._branch_first_node: Mapping[str, NodeID] = MappingProxyType(
    graph.get_branch_first_nodes()  # Returns dict[str, NodeID]
)
```

**Step 3b — Simplified `create_continuation_work_item` (single code path):**

```python
def create_continuation_work_item(self, *, token, current_node_id, coalesce_name=None, ...):
    if coalesce_name is not None:
        coalesce_node_id = self._coalesce_node_ids[coalesce_name]
        first_node = self._branch_first_node[token.branch_name]  # Direct access, no .get()
        return self.create_work_item(
            token=token,
            current_node_id=first_node,
            coalesce_name=coalesce_name,
            coalesce_node_id=coalesce_node_id,
        )
    # ... normal continuation
```

For identity branches, `first_node == coalesce_node_id`, preserving current behavior.
For transform branches, `first_node` is the first transform in the chain.

**Step 3c — `DAGTraversalContext` extension:**

Extend the frozen `DAGTraversalContext` dataclass with a `branch_first_node` field,
consistent with how `coalesce_node_map` is already plumbed from orchestrator to
navigator.

**Step 3d — Ensure `coalesce_name` propagates through work items:**

The work item must carry `coalesce_name` and `coalesce_node_id` through the branch
transform chain. When the processor finishes a branch transform, the next work item
(from `node_to_next`) inherits these fields. When the token arrives at the coalesce
node (`current_node_id == coalesce_node_id`), the existing coalesce detection fires.

### Layer 4: Execution Graph Metadata

**`ExecutionGraph.get_branch_first_nodes()`** — NEW method:

Returns `dict[str, NodeID]` mapping every branch name to its first processing node:
- Identity branches → coalesce node ID
- Transform branches → first transform's node ID

Built from graph topology by examining coalesce branch configs and tracing the
connection chain from each fork branch.

---

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| **No transforms on any branch** | List format: identical to current behavior. Dict with identity mappings: same edges, same routing. |
| **Transforms on some branches only** | Mixed: transformed branches use connection system, direct branches use COPY edges. |
| **Chained transforms on one branch** | Standard connection resolution: `input`/`on_success` chain. `node_to_next` traverses the chain. |
| **Transform fails on one branch** | Token gets error outcome. `notify_branch_lost()` fires for that branch. Merge policy decides (e.g., `require_all` fails, `best_effort` proceeds). |
| **Transform is an aggregation** | NOT SUPPORTED. Aggregation on a fork branch breaks the 1:1 token→branch invariant. Reject at DAG validation. |
| **Nested fork inside a branch** | NOT SUPPORTED. A transform on a branch cannot be a fork gate. Reject at DAG validation. |
| **Same transform consuming from multiple branches** | Not possible: a transform has one `input`. Each branch needs its own transform instance. |
| **Branch transform produces to a sink** | Valid: `on_error: <sink>` routes errors to a sink. `on_success` must continue to the coalesce input connection. |
| **Intermediate connection name = a branch name** | Caught by connection namespace validation (branch names reserved when registered as producers). |
| **Branch transform produces to a different coalesce** | Caught by connection resolution — consumer mismatch. |
| **All branches identity-mapped in dict format** | Equivalent to list format. Identity branches get direct COPY edges. |
| **`on_success` on coalesce with per-branch transforms** | Works unchanged — `on_success` applies after merge, not per-branch. |
| **Very long branch transform chains (10+)** | Works but increases audit trail depth. Natural governor: config readability. |

---

## Validation Rules (New)

1. **Branch identity uniqueness**: Dict keys in `branches` must be globally unique
   across all coalesce nodes (existing rule, unchanged).
2. **Branch input connections**: Dict values must either (a) match a fork branch name
   (identity mapping) or (b) be a valid connection produced by some transform.
3. **No aggregation on branches**: If a transform in a branch chain is an aggregation
   type, reject at DAG construction. Aggregation breaks 1:1 token identity.
4. **No nested forks**: If a transform in a branch chain is a fork gate, reject at DAG
   construction. Nested forks require a separate design.
5. **Every fork branch must reach coalesce**: All branches declared in `fork_to` must
   either (a) map to a coalesce branch or (b) route to a sink. No dangling branches.
6. **Schema compatibility by strategy**: Union requires compatible overlapping types.
   Nested and select have no cross-branch schema constraint.
7. **Connection namespace isolation**: Branch names used as connections must not collide
   with existing transform connection names.

---

## Test Strategy

### Unit Tests

| Test | What It Validates |
|------|-------------------|
| Config parsing: list branches | Backward compat — list normalizes to identity dict |
| Config parsing: dict branches | New format parses correctly |
| Config parsing: list↔dict equivalence | `branches: [a, b]` produces same result as `branches: {a: a, b: b}` |
| Config validation: invalid branch names/values | Rejects reserved names, duplicates, invalid connection names |
| Config validation: dict with non-existent connection value | Rejected — no producer for connection |
| Builder: single-transform branch | Fork → transform → coalesce edges created correctly |
| Builder: chained transforms on branch | Multi-hop chain wired correctly |
| Builder: mixed (some branches with transforms, some without) | Direct COPY edges for identity branches, connection-system edges for transform branches |
| Builder: aggregation on branch rejected | Validation error at DAG construction |
| Builder: nested fork on branch rejected | Validation error at DAG construction |
| Builder: connection namespace collision (branch name = transform connection) | Rejected |
| Builder: branch transform producing to wrong coalesce | Rejected |
| Schema validation: union with compatible types | Passes |
| Schema validation: union with incompatible types | Rejected |
| Schema validation: nested with different schemas | Passes |
| `get_branch_first_nodes()` correctness | Returns correct first node for each branch |
| `get_branch_first_nodes()` identity branches | Maps to coalesce node ID |
| Navigator: branch routing through transforms | Fork children routed to first transform |
| Navigator: identity branch routing | Fork children go directly to coalesce (unchanged) |
| WorkItem: `coalesce_name` propagation through branch chain | Preserved through transform chain |
| Processor: coalesce detection after final branch transform | Token coalesces after traversing branch |
| `node_to_next` map: branch transform chains included | Correct `node_to_next` entries for branch transforms |

### Integration Tests

| Test | What It Validates |
|------|-------------------|
| Fork → per-branch transforms → coalesce (require_all, nested) | End-to-end data flow, correct merge |
| Fork → per-branch transforms → coalesce (union merge) | Field combination from different transforms |
| Fork → mixed branches (one with transforms, one direct) | Both paths arrive at coalesce correctly |
| Branch transform failure → branch_lost → merge policy | `best_effort` proceeds, `require_all` fails |
| Branch transform `on_error` routing within fork | Errors on branch routed to correct sink |
| Per-branch transforms with schema contracts | Guaranteed fields propagate correctly |
| Retry of failed branch transform | Preserves coalesce context after retry |
| Checkpoint + resume with per-branch transforms | Interrupted mid-branch resumes correctly |
| Union merge with field collisions from different branch transforms | Collision tracking recorded in audit trail |

### Property Tests

| Test | What It Validates |
|------|-------------------|
| Arbitrary branch counts (2-10) with random transform chains | DAG always validates, tokens always reach coalesce or fail explicitly |
| Random merge policy × strategy combinations | All policy/strategy combos work with per-branch transforms |
| Config normalization equivalence | list format ≡ identity dict for all valid branch configs |
| Token identity preservation through branch transforms | `branch_name` unchanged after N transforms |
| Connection registration invariants | All registered producers consumed, no dangling outputs |

### E2E Tests

| Test | What It Validates |
|------|-------------------|
| Full YAML → CSV pipeline with per-branch transforms | Real config → real execution → correct output |

### Existing Factories to Reuse

| Factory | Location | Use For |
|---------|----------|---------|
| `make_graph_fork(branches={"path_a": ["t1"], ...})` | `tests/fixtures/factories.py:160` | DAG construction unit tests |
| `fork_with_branch_transforms()` Hypothesis strategy | `tests/property/core/test_dag_complex_topologies.py:159` | Property tests (already exists!) |
| `_settings(branches=...)` helper | `tests/unit/engine/test_coalesce_executor.py:114` | Coalesce executor tests |
| `_make_token(branch_name=...)` helper | `tests/unit/engine/test_coalesce_executor.py:55` | Token creation |
| `wire_transforms()` | `tests/fixtures/factories.py:216` | Transform chain wiring |

### Tests Likely to Need Updating

| Test | Reason |
|------|--------|
| `test_dag_navigator.py::test_jumps_to_coalesce_when_name_provided` | Tests direct-jump behavior being modified |
| `test_coalesce_executor.py` — `_settings()` calls | Verify normalization doesn't break list-format usage |
| `test_config.py` — CoalesceSettings tests | Add dict-format companions |
| `test_dag.py` — `TestCoalesceGraph` | Add per-branch transform variant |

### Example Validation

Add `examples/fork_coalesce/settings_per_branch.yaml` demonstrating per-branch
transforms using the new dict syntax. This example is the primary teaching tool for
the feature (the list→dict "format cliff" means examples drive adoption).

---

## Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `core/config.py` | MODIFY | `CoalesceSettings.branches` → `dict[str, str]` with `field_validator(mode="before")` for list normalization; update `validate_branch_names` for dict |
| `core/dag/builder.py` | MODIFY | Conditionally register fork branches as connections; split edge creation by transform presence; strategy-aware schema validation |
| `core/dag/graph.py` | MODIFY | `get_branch_first_nodes()` method; strategy-aware `validate_edge_compatibility`; strategy-aware schema population |
| `engine/dag_navigator.py` | MODIFY | `_branch_first_node: Mapping[str, NodeID]` via `MappingProxyType`; simplified `create_continuation_work_item()` (single code path, direct access) |
| `engine/processor.py` | VERIFY | Ensure `coalesce_name` propagates through work item chain (likely no changes) |
| `engine/orchestrator/core.py` | MODIFY | Extend `DAGTraversalContext` with `branch_first_node`; pass to navigator |
| `examples/fork_coalesce/` | ADD | `settings_per_branch.yaml` variant |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Connection namespace collision (branch name = existing connection) | Validate at builder time; only register branches that have transforms |
| `node_to_next` ambiguity (coalesce has multiple predecessors) | Each branch chain has its own `node_to_next` entries; processor uses `coalesce_node_id` from work item |
| Breaking existing fork/coalesce configs | List format normalized to identity dict; zero behavior change for existing configs |
| Checkpoint recovery with mid-branch interruption | Child tokens have individual outcomes; resume replays incomplete tokens. Stateless transforms are idempotent. |
| Performance regression from connection resolution | O(branches × transforms) — negligible for realistic DAG sizes |
| Mixed COPY/MOVE edges on coalesce node | Schema population made strategy-aware (not dependent on edge iteration order) |
| Convoy effect (slow branch blocks fast branches) | Existing `timeout_seconds` + `best_effort`/`first` policies are the natural governor |

---

## Systems Analysis Summary

**Archetype:** Limits to Growth — three barriers are the constraining process; this
design removes them.

**No new feedback loops created.** Change operates on construction-time topology, not
runtime adaptive behavior. Existing balancing loops (coalesce timeout, schema
validation) are preserved and strengthened.

**Primary runtime effect:** Convoy effect under `require_all` — slowest branch
determines overall latency. Natural governor: `timeout_seconds` with `best_effort`.

**Predicted emergent patterns:**
- Hedged requests (`first` policy + per-branch transforms for latency optimization)
- Progressive enrichment chains (mini-pipelines within branches)
- Asymmetric branch configurations (one heavy branch, one light)

**Follow-on work (predicted):** Users will request nested forks, aggregation on
branches, and cross-branch communication within 1-2 releases. These are correctly
scoped out but should be planned for.

---

## Out of Scope

- **Aggregation on fork branches**: Breaks 1:1 token→branch invariant. Requires separate design.
- **Nested forks**: Fork inside a fork branch. Requires recursive branch scoping.
- **Dynamic branch count**: Runtime-determined number of branches. Requires generator pattern.
- **Cross-branch communication**: Transforms on branch A reading data from branch B.

---

## Review History

| Date | Reviewer | Key Findings |
|------|----------|-------------|
| 2026-02-13 | Architecture Critic | DAGTraversalContext gap (HIGH); mixed edge modalities; dual schema validation paths; dangling producer bug |
| 2026-02-13 | Systems Thinker | Limits to Growth archetype; convoy effect risk; no new feedback loops; Growth and Underinvestment prediction |
| 2026-02-13 | Python Reviewer | RuntimeCoalesceConfig doesn't exist (CRITICAL); wrong Pydantic pattern; defensive `.get()` violation |
| 2026-02-13 | Test Analyst | 7 critical missing tests; 14 high/medium gaps; existing `fork_with_branch_transforms()` factory reusable |
