# ELSPETH System Operations Contract

> **Status:** FINAL (v1.2)
> **Last Updated:** 2026-02-13
> **Authority:** This document is the master reference for all engine-level system operations.
> **Companion:** [Plugin Protocol Contract](plugin-protocol.md) covers Source, Transform, and Sink plugins.

## Overview

System operations are **engine-level infrastructure** that controls token flow through the DAG. Unlike plugins (which touch row contents and require code), system operations work on **wrapped data** — tokens, routing metadata, and merge buffers — and are driven entirely by configuration.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      DATA FLOW ARCHITECTURE                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PLUGINS (touch row contents)                                        │
│  ─────────────────────────────                                       │
│  Source ──► [row data] ──► Transform ──► [row data] ──► Sink         │
│                                                                      │
│  SYSTEM OPERATIONS (touch token metadata)                            │
│  ────────────────────────────────────────                            │
│  Gate ──► "where does this token go?"                                │
│  Fork ──► "copy token to parallel paths"                             │
│  Coalesce ──► "merge tokens from parallel paths"                     │
│  Aggregation ──► "batch tokens until trigger fires"                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Why These Are Not Plugins

1. **They don't touch row contents** — A gate evaluates a condition but need not modify the row. Fork copies tokens. Coalesce combines branch outputs per config strategy.
2. **They require DAG coordination** — Fork/coalesce semantics span multiple nodes. The engine must track token lineage, manage parallel paths, and handle timeouts.
3. **Config is sufficient** — All behavior is expressible via expressions, policies, and strategies.
4. **100% our code** — These are ELSPETH internals. No user extension points, no defensive programming needed. Bugs crash immediately.

---

## Routing Primitives

Before describing individual operations, these are the foundational routing types shared across all system operations.

### RoutingKind

```python
class RoutingKind(StrEnum):
    CONTINUE = "continue"           # Continue to next node in pipeline
    ROUTE = "route"                 # Route to labeled destination (sink)
    FORK_TO_PATHS = "fork_to_paths" # Fork to multiple parallel paths
```

### RoutingMode

```python
class RoutingMode(StrEnum):
    MOVE = "move"      # Token exits current path, goes to destination only
    COPY = "copy"      # Token clones to destination AND continues (fork only)
    DIVERT = "divert"  # Token diverted to error/quarantine sink (structural)
```

### RoutingAction

```python
@dataclass(frozen=True)
class RoutingAction:
    kind: RoutingKind
    destinations: tuple[str, ...]   # Sink names or branch names
    mode: RoutingMode
    reason: RoutingReason | None = None

    # Factory methods:
    @classmethod
    def continue_(cls, *, reason=None) -> RoutingAction
    @classmethod
    def route(cls, label: str, *, mode=RoutingMode.MOVE, reason=None) -> RoutingAction
    @classmethod
    def fork_to_paths(cls, paths: list[str], *, reason=None) -> RoutingAction
```

**Invariants** (enforced by `__post_init__`):

| Kind | Destinations | Mode | Constraint |
|------|-------------|------|------------|
| `CONTINUE` | Empty | `MOVE` (COPY forbidden) | No destination needed |
| `ROUTE` | Exactly one | `MOVE` (COPY forbidden) | Use `fork_to_paths` for multi-destination |
| `FORK_TO_PATHS` | One or more | `COPY` only | Path names must be unique, list must not be empty |

### RoutingReason (Audit Metadata)

Every routing decision carries a structured reason for the audit trail:

```python
# Config gate
class ConfigGateReason(TypedDict):
    condition: str      # The expression evaluated
    result: str         # The route label produced

# Transform error routing
class TransformErrorReason(TypedDict):
    reason: str         # Error category string

# Source quarantine routing
class SourceQuarantineReason(TypedDict):
    quarantine_error: str   # Description of the validation failure

# Discriminated union — field presence distinguishes variants
RoutingReason = ConfigGateReason | TransformErrorReason | SourceQuarantineReason
```

---

## Gates

Gates make **routing decisions**. They evaluate a condition on a token's row data and decide where the token goes next. All gates are **config-driven** using a restricted expression parser built on Python's AST — never `eval()`.

### Config Expression Gates

#### Configuration

```yaml
gates:
  - name: quality_check
    condition: "row['confidence'] >= 0.85"
    routes:
      "true": continue          # Continue to next node
      "false": review_sink      # Route to named sink

  - name: tier_router
    condition: "row['amount'] > 2500 and 'premium' or (row['amount'] > 1000 and 'high' or 'normal')"
    routes:
      premium: premium_sink
      high: high_value_sink
      normal: continue
```

#### Evaluation Flow

```
Token arrives at config gate
    │
    ▼
ExpressionParser(condition)
    │
    ├── Phase 1: ast.parse() — parse expression into AST
    ├── Phase 2: _ExpressionValidator — reject forbidden constructs
    └── Phase 3: _ExpressionEvaluator — evaluate against row
    │
    ▼
Result conversion:
    bool  → "true" / "false"
    str   → use as route label directly
    other → str(result)
    │
    ▼
Route resolution:
    Look up label in gate.routes
    → "continue": token continues to next node
    → "fork": create child tokens (see Fork section)
    → sink_name: token routed to sink
```

#### Expression Language Reference

**Allowed constructs:**

| Construct | Example | Notes |
|-----------|---------|-------|
| Field access | `row['field']`, `row.get('field')` | PipelineRow supports dual-name access |
| Comparisons | `==`, `!=`, `<`, `>`, `<=`, `>=` | Standard Python semantics |
| Boolean operators | `and`, `or`, `not` | Short-circuit evaluation |
| Identity | `is`, `is not` | For `None` checks |
| Membership | `in`, `not in` | Lists, dicts, strings |
| Ternary | `x if condition else y` | Conditional expressions |
| Arithmetic | `+`, `-`, `*`, `/`, `//`, `%` | For computed routing |
| Literals | strings, numbers, booleans, `None` | Immutable values |
| Collections | `[1, 2, 3]`, `{'a': 1}`, `(1, 2)`, `{1, 2}` | Lists, dicts, tuples, sets for membership checks |

**Forbidden constructs (rejected at parse time):**

| Construct | Why Forbidden |
|-----------|---------------|
| Lambda expressions | Code injection vector |
| Comprehensions | Arbitrary iteration |
| Assignment (`:=`) | Side effects |
| `await`, `yield` | Control flow hijacking |
| F-strings with expressions | Arbitrary code execution |
| Attribute access (except `row.get`) | Object traversal |
| Names other than `row`, `True`, `False`, `None` | Scope escape |
| Function calls (except `row.get()`) | Arbitrary execution |
| Imports | Module system access |
| Starred expressions (`*x`, `**x`) | Unpacking abuse |
| Slice syntax (`[1:3]`) | Index manipulation |

An expression like `"__import__('os').system('rm -rf /')"` is rejected at config validation time, not at runtime.

#### Expression Error Handling

| Error Type | When | Behavior |
|------------|------|----------|
| `ExpressionSyntaxError` | Expression is not valid Python syntax | Pipeline fails at config validation (before any rows) |
| `ExpressionSecurityError` | Forbidden construct in expression | Pipeline fails at config validation (before any rows) |
| `ExpressionEvaluationError` | Runtime error (KeyError, ZeroDivisionError, TypeError) | Node state recorded as FAILED, exception propagates to orchestrator which fails the run (row data caused the error in OUR expression — this is a config bug) |

### Gate Execution Contract

1. **Record node state start** — input hash computed, `begin_node_state()` called
2. **Execute gate** — expression evaluation
3. **Populate audit fields** — output hash, duration
4. **Process routing** — based on `RoutingAction.kind`
5. **Complete node state** — always `COMPLETED` for successful execution
6. **Record routing event(s)** — one per destination, with edge ID and mode

**Terminal state derivation:** Gate node states are always `COMPLETED`. The token's terminal state (`ROUTED`, `FORKED`) is derived from the routing events, not stored in `node_states.status`.

### Gate Audit Trail

| Record | Contents |
|--------|----------|
| `node_states` | `state_id`, `token_id`, `node_id`, `status=COMPLETED`, `input_hash`, `output_hash`, `duration_ms` |
| `routing_events` | `state_id`, `edge_id`, `mode` (MOVE/COPY), `reason` (condition text or plugin metadata) |
| Row modifications | If gate modified the row: `input_hash ≠ output_hash`, delta traceable |

---

## Fork (Token Splitting)

Fork creates **child tokens** from a parent token for parallel processing across multiple DAG paths.

### How Forks Are Triggered

Forks are triggered by gates. A gate returns `RoutingAction.fork_to_paths(["path_a", "path_b", ...])` and the engine creates child tokens.

```yaml
gates:
  - name: parallel_analysis
    condition: "True"            # Always fork
    routes:
      "true": fork               # Special keyword triggers fork
    fork_to:
      - sentiment_path
      - entity_path
      - summary_path

# Parallel paths defined separately
paths:
  sentiment_path:
    - transform: sentiment_analyzer
  entity_path:
    - transform: entity_extractor
  summary_path:
    - transform: summarizer
```

### Fork Execution Contract

```
Gate returns RoutingAction.fork_to_paths(["path_a", "path_b"])
    │
    ▼
GateExecutor creates fork_row (PipelineRow with contract)
    │
    ▼
TokenManager.fork_token(parent_token, branches, step_in_pipeline, run_id, row_data=None)
    │
    ├── LandscapeRecorder.fork_token()  ← ATOMIC OPERATION
    │   ├── Creates all child token records in audit DB
    │   ├── Records parent token outcome: FORKED
    │   └── Returns children with fork_group_id
    │
    ├── For each child:
    │   ├── copy.deepcopy(row_data)  ← CRITICAL: prevents sibling mutation
    │   ├── Set branch_name from child.branch_name
    │   └── Set fork_group_id from recorder result
    │
    └── Returns (child_infos: list[TokenInfo], fork_group_id: str)
```

### Token Lineage After Fork

```
Parent Token (T1)
    │ row_id: R1
    │ token_id: T1
    │ outcome: FORKED (terminal)
    │
    ├──► Child Token (T2)
    │    row_id: R1              ← Same source row
    │    token_id: T2            ← Unique instance
    │    branch_name: "sentiment_path"
    │    fork_group_id: FG1      ← Groups siblings
    │
    ├──► Child Token (T3)
    │    row_id: R1
    │    token_id: T3
    │    branch_name: "entity_path"
    │    fork_group_id: FG1
    │
    └──► Child Token (T4)
         row_id: R1
         token_id: T4
         branch_name: "summary_path"
         fork_group_id: FG1
```

### Fork Invariants

1. **Atomicity** — Child token creation and parent `FORKED` outcome are recorded in a single database operation. No partial forks.
2. **Deep copy** — Each child gets an independent `deepcopy` of the row data. Without this, mutations in one branch (e.g., adding a field) would leak to sibling branches via shared mutable objects.
3. **Contract preservation** — All children share the same `SchemaContract` reference (contracts are immutable frozen dataclasses, safe to share).
4. **Parent is terminal** — The parent token reaches `FORKED` and never processes further. Only children continue.
5. **Branch names are unique** — Enforced by `RoutingAction.__post_init__()`.
6. **All branches must be wired** — If a gate produces a branch name not connected to a coalesce or sink, DAG validation fails at construction time.

### Fork Audit Trail

| Record | Contents |
|--------|----------|
| `tokens` | One record per child: `token_id`, `row_id`, `fork_group_id` |
| `token_outcomes` | Parent: `outcome=FORKED`. Children: outcomes determined by their individual journeys |
| `routing_events` | One per branch: `state_id`, `edge_id`, `mode=COPY` |

---

## Coalesce (Token Merging)

Coalesce is the join barrier for fork/join patterns. It waits for tokens from specified branches, then merges them into a single token based on a configurable policy and strategy.

### Configuration

```yaml
coalesce:
  - name: merge_results
    branches:
      - sentiment_path
      - entity_path
      - summary_path
    policy: require_all       # Wait for all branches
    timeout_seconds: 300      # Max wait (5 minutes)
    merge: union              # How to combine row data
```

### Merge Policies

Policies determine **when** to merge.

| Policy | Fires When | On Timeout | On Branch Loss |
|--------|------------|------------|----------------|
| `require_all` | ALL branches arrive | FAIL all pending tokens | FAIL immediately |
| `quorum` | N branches arrive (`quorum_count`) | FAIL if quorum not met, MERGE if met | FAIL if quorum impossible, MERGE if already met |
| `best_effort` | All branches accounted for (arrived + lost) | MERGE whatever arrived | MERGE if all accounted for |
| `first` | ANY branch arrives (first one) | N/A (merges immediately) | N/A |

### Merge Strategies

Strategies determine **how** to combine row data from branches.

#### Union (Default)

Combine all fields from all branches. Last writer wins on collisions.

```
Branch A: {id: 1, sentiment: "positive"}
Branch B: {id: 1, entities: ["ACME", "NYC"]}
Branch C: {id: 1, summary: "..."}

Merged:   {id: 1, sentiment: "positive", entities: ["ACME", "NYC"], summary: "..."}
          ↑ "id" appears in all branches — last writer (C) wins
```

**Collision tracking:** Union collisions are recorded in the audit trail metadata so you can trace which branch's value was kept.

#### Nested

Each branch output becomes a nested object keyed by branch name.

```
Branch A: {sentiment: "positive"}
Branch B: {entities: ["ACME", "NYC"]}

Merged:   {sentiment_path: {sentiment: "positive"},
           entity_path: {entities: ["ACME", "NYC"]}}
```

#### Select

Take output from a specific branch only, discard others.

```yaml
coalesce:
  - name: primary_result
    branches: [fast_path, slow_path]
    policy: first
    merge: select
    select_branch: fast_path
```

### Coalesce Execution Contract

```
Token arrives at coalesce point
    │
    ▼
CoalesceExecutor.accept(token, coalesce_name, step)
    │
    ├── Is (coalesce_name, row_id) already completed?
    │   YES → Record late arrival failure, return
    │   NO  → Continue
    │
    ├── Create or retrieve _PendingCoalesce for (coalesce_name, row_id)
    │
    ├── Validate branch_name is in expected branches
    │
    ├── Record token arrival and begin_node_state(PENDING)
    │
    ├── Evaluate merge condition via _should_merge(policy, pending)
    │   │
    │   ├── require_all:  arrived_count == expected_count
    │   ├── first:        arrived_count >= 1
    │   ├── quorum:       arrived_count >= quorum_count
    │   └── best_effort:  (arrived + lost) >= expected_count
    │
    ├── Condition NOT met → return CoalesceOutcome(held=True)
    │
    └── Condition met → _execute_merge()
        │
        ├── Merge row data per strategy (union/nested/select)
        ├── Build merged SchemaContract
        │   ├── union:  SchemaContract.merge() across branches
        │   ├── nested: new contract with branch keys as object fields
        │   └── select: use selected branch's contract directly
        ├── TokenManager.coalesce_tokens(parents, merged_data)
        ├── Record COALESCED outcome for all consumed tokens
        ├── Complete all pending node_states with COMPLETED
        ├── Mark key as completed (for late arrival detection)
        └── Return CoalesceOutcome(merged_token=...)
```

### Internal Data Structures

```python
@dataclass
class CoalesceOutcome:
    held: bool                                  # True if waiting for more branches
    merged_token: TokenInfo | None = None       # The merged result
    consumed_tokens: list[TokenInfo]             # Tokens that were consumed
    coalesce_metadata: dict[str, Any] | None     # Audit info (policy, timing)
    failure_reason: str | None = None            # If merge failed
    coalesce_name: str | None = None
    outcomes_recorded: bool = False              # Whether COALESCED outcomes already written

@dataclass
class _PendingCoalesce:
    arrived: dict[str, TokenInfo]               # branch_name → token
    arrival_times: dict[str, float]             # branch_name → monotonic time
    first_arrival: float                        # For timeout calculation
    pending_state_ids: dict[str, str]           # branch_name → state_id
    lost_branches: dict[str, str]               # branch_name → reason
```

### Timeout Handling

Timeouts are checked at specific points during processing (not via background timers):

```
check_timeouts(coalesce_name)
    │
    ├── For each pending (coalesce_name, row_id):
    │   elapsed = now - first_arrival
    │   if elapsed >= timeout_seconds:
    │       process timeout per policy
    │
    ├── require_all → FAIL all pending tokens
    ├── quorum → FAIL if threshold not met, MERGE if met
    └── best_effort → MERGE whatever arrived
```

**Known Limitation (True Idle):** Timeout checks fire when the next token arrives at the coalesce point or when the source completes. During completely idle periods with no data flowing, timeouts cannot fire. For streaming sources, combine coalesce timeouts with source-level heartbeat rows.

### Late Arrivals

A token arriving after its coalesce group has already merged:

```
Token T5 (branch: entity_path) arrives at coalesce
    │
    ├── Check: Is (merge_results, R1) in _completed_keys?
    │   YES → This is a late arrival
    │
    ├── Record failure in audit trail:
    │   failure_reason = "late_arrival_after_merge"
    │
    └── Return CoalesceOutcome with failure_reason
        (token reaches FAILED terminal state)
```

**Memory management:** `_completed_keys` is a FIFO-bounded `OrderedDict` capped at 10,000 entries. After eviction, a late arrival for an ancient row_id creates a new pending entry rather than being detected as late. This is a deliberate trade-off: bounded memory vs perfect late-arrival detection for very old rows.

### Branch Loss Notification

When an upstream error routes a token to an error sink instead of the coalesce, the engine notifies the coalesce that the branch will never arrive:

```python
def notify_branch_lost(
    self,
    coalesce_name: str,
    row_id: str,
    lost_branch: str,
    reason: str,
    step_in_pipeline: int,
) -> CoalesceOutcome | None
```

**Policy-specific consequences of branch loss:**

| Policy | On Branch Loss |
|--------|---------------|
| `require_all` | Immediate failure — cannot satisfy "all" |
| `quorum` | Fail if quorum now impossible (remaining + arrived < threshold). Merge if already met. |
| `best_effort` | Merge if all branches now accounted for (arrived + lost = expected) |
| `first` | No action (should already have merged on first arrival) |

### Token Lineage After Coalesce

```
Child Tokens (arriving from branches)
    │
    ├── T2 (sentiment_path): {sentiment: "positive"}
    │   outcome: COALESCED
    ├── T3 (entity_path): {entities: ["ACME"]}
    │   outcome: COALESCED
    └── T4 (summary_path): {summary: "..."}
        outcome: COALESCED
    │
    ▼ (coalesce with union strategy)
    │
Merged Token (T5)
    row_id: R1
    token_id: T5 (new)
    join_group_id: JG1
    row_data: {sentiment: "positive", entities: ["ACME"], summary: "..."}
    │
    ▼ continues through pipeline...
```

### Coalesce Invariants

1. **Correlation by row_id** — Tokens from the same source row (same `row_id`) are matched across branches. Different source rows never merge.
2. **Branch uniqueness** — A branch can only contribute one token per row_id. Duplicate arrivals for the same branch raise an error.
3. **Consumed tokens are terminal** — All tokens consumed in a merge reach `COALESCED` terminal state.
4. **Merged token is new** — The merged token gets a fresh `token_id` with `join_group_id` linking back to consumed tokens.
5. **Schema merge follows mode precedence** — When merging contracts: `FIXED > FLEXIBLE > OBSERVED`. The strictest mode wins.
6. **End-of-source flush** — When the source exhausts, all pending coalesces are flushed per their policy. `best_effort` merges what arrived; `require_all` fails remaining.

### Coalesce Audit Trail

| Record | Contents |
|--------|----------|
| `node_states` | One per arriving branch (PENDING until merge, then COMPLETED). One for merged output. |
| `token_outcomes` | Consumed tokens: `outcome=COALESCED`. Merged token: outcome determined by downstream journey. |
| `routing_events` | Edge from coalesce node to next node (MOVE mode) |
| Metadata | `policy`, `branches_expected`, `branches_arrived`, `arrival_order`, `wait_duration_ms`, `union_collisions` (if any) |

---

## Aggregation (Token Batching)

Aggregation collects multiple tokens until a trigger fires, then processes them as a batch. The engine owns the buffer — transforms simply receive `list[dict]` and return a result.

### Configuration

```yaml
aggregations:
  - node_id: batch_stats
    trigger:
      count: 100                    # Fire after 100 rows
      timeout_seconds: 3600         # Or after 1 hour
    output_mode: transform          # N inputs → M outputs (default)
    expected_output_count: 1        # Optional cardinality validation

transforms:
  - plugin: summary_transform
    node_id: batch_stats            # Must match aggregation node_id
    # Transform must have is_batch_aware = True
```

### Output Modes

| Mode | Input → Output | Token Handling | While Buffering | Use Case |
|------|----------------|----------------|-----------------|----------|
| `transform` | N → M | New tokens created via `expand_token()` | Input tokens: `CONSUMED_IN_BATCH` (terminal) | Classic aggregation (sum, count, mean) |
| `passthrough` | N → N | Same tokens preserved | Tokens: `BUFFERED` (non-terminal) | Batch enrichment |

### Trigger Types

| Trigger | Fires When | Combined Behavior |
|---------|------------|-------------------|
| `count` | N tokens accumulated | First trigger to fire wins |
| `timeout_seconds` | Duration elapsed since batch start | Checked before each row |
| `condition` | Row matches expression | Immediate flush |
| `manual` | Explicitly triggered via API/CLI | On-demand flush |
| End-of-source | Source exhausted | Always checked (implicit) |

### Aggregation Execution Flow

```
Token T1 arrives at aggregation node
    │
    ▼
AggregationExecutor.accept(token, node_id)
    │
    ├── Buffer row_data internally
    ├── Record batch membership in audit trail
    ├── Token outcome: CONSUMED_IN_BATCH (transform mode)
    │                  or BUFFERED (passthrough mode)
    │
    ├── Evaluate triggers:
    │   count >= threshold?     → FLUSH
    │   timeout elapsed?        → FLUSH
    │   condition matches row?  → FLUSH
    │   source exhausted?       → FLUSH
    │
    ├── No trigger → return (token held)
    │
    └── Trigger fires → execute_flush()
        │
        ├── Batch state: draft → executing
        ├── Retrieve buffered rows as list[dict]
        ├── Call transform.process(rows, ctx)
        ├── Batch state: executing → completed
        │
        ├── transform mode:
        │   └── Create new tokens via expand_token()
        │       (parent linkage to triggering token)
        │
        └── passthrough mode:
            └── Enriched rows continue with original token IDs
                (BUFFERED → COMPLETED on flush)
```

### Crash Recovery

Aggregation buffers are persisted in checkpoints:

- `get_checkpoint_state()` serializes buffered rows and batch metadata
- `restore_from_checkpoint()` restores buffers after crash
- Trigger evaluators resume from correct count
- In-progress batches survive crashes and can be resumed

### Timeout Behavior

Timeout triggers are checked **before** each row is processed, not via background timers.

**Known Limitation (True Idle):** If no rows arrive, buffered data will not flush until either:
1. A new row arrives (triggering the timeout check)
2. The source completes (triggering end-of-source flush)

For streaming sources that may never end, combine timeout with count triggers, or implement periodic heartbeat rows at the source level.

### Aggregation Invariants

1. **Engine owns the buffer** — Transforms never manage batch state. This enables crash recovery, consistent trigger evaluation, and clean audit trail.
2. **Atomic batch execution** — If the transform returns `error`, ALL buffered rows fail together.
3. **Cardinality validation** — If `expected_output_count` is set and the transform returns a different count, the batch fails.
4. **Passthrough preserves token identity** — In `passthrough` mode, the same `token_id` values continue after enrichment.
5. **Transform mode creates new lineage** — New tokens are created via `expand_token()` with parent linkage.

### Aggregation Audit Trail

| Record | Contents |
|--------|----------|
| `batches` | `batch_id`, `trigger_type`, `aggregation_node_id`, `status` (draft/executing/completed/failed) |
| `batch_members` | Which tokens belong to which batch (ordinal position) |
| `node_states` | Transform input/output hashes for the batch call |
| `token_outcomes` | Input tokens: `CONSUMED_IN_BATCH` or `BUFFERED`. Output tokens: determined by downstream journey. |

---

## Token Identity Through Flow Control

Every token carries identity and lineage metadata that tracks its journey through system operations.

### TokenInfo Contract

```python
@dataclass(frozen=True, slots=True)
class TokenInfo:
    row_id: str                         # Stable source row identity
    token_id: str                       # Instance of row in specific DAG path
    row_data: PipelineRow               # Data with schema contract

    # Flow control lineage
    branch_name: str | None = None      # Fork path identifier
    fork_group_id: str | None = None    # Groups siblings from same fork
    join_group_id: str | None = None    # Groups tokens merged in coalesce
    expand_group_id: str | None = None  # Groups children from deaggregation

    def with_updated_data(self, new_data: PipelineRow) -> TokenInfo:
        """Return new TokenInfo with updated row_data, preserving all lineage."""
```

### Identity Semantics

| Field | Stable Across | Changes When | Purpose |
|-------|--------------|--------------|---------|
| `row_id` | Fork, coalesce, aggregation | Never (source row identity) | Correlating tokens from same source row |
| `token_id` | Transform, gate | Fork (new children), coalesce (new merged), expand (new children) | Unique instance in specific DAG path |
| `branch_name` | Within a branch | Set at fork, cleared at coalesce | Identifying which branch a token is on |
| `fork_group_id` | Within a branch | Set at fork | Grouping siblings from same fork |
| `join_group_id` | After coalesce | Set at coalesce | Linking merged token to consumed tokens |
| `expand_group_id` | After expand | Set at expand | Linking expanded children to parent |

### Terminal States from Flow Operations

| Operation | Parent Token State | Child Token State |
|-----------|-------------------|-------------------|
| **Fork** | `FORKED` (terminal) | Determined by downstream journey |
| **Coalesce** | `COALESCED` (terminal) for consumed tokens | Merged token continues |
| **Aggregation (transform mode)** | `CONSUMED_IN_BATCH` (terminal) | New tokens via `expand_token()` |
| **Aggregation (passthrough mode)** | `BUFFERED` (non-terminal) → `COMPLETED` on flush | Same tokens continue |
| **Deaggregation** | `EXPANDED` (terminal) | New tokens via `expand_token()` |

### Complete Token State Diagram

```
                         Source creates token
                                │
               ┌────────────────┼──────────────────┐
               │                │                  │
               ▼                ▼                  ▼
        ┌─────────────┐  ┌──────────┐       ┌───────────┐
        │ QUARANTINED │  │ CREATED  │       │(valid row) │
        └─────────────┘  │(invalid) │       └─────┬─────┘
        (source validation └──────────┘             │
         failure)                    ┌──────────────┼───────────────────┐
                                     │              │                   │
                                     ▼              ▼                   ▼
                                ┌────────┐    ┌──────────┐       ┌───────────┐
                                │ FORKED │    │ BUFFERED │       │ Processing│
                                └────┬───┘    └────┬─────┘       └─────┬─────┘
                                     │             │                   │
                                (children)    (on flush)    ┌──────────┼──────────┬───────────┐
                                                   │        │          │          │           │
                                                   ▼        ▼          ▼          ▼           ▼
                                             ┌─────────┐ ┌──────┐ ┌────────┐ ┌────────────┐ ┌─────────┐
                                             │COMPLETED│ │ROUTED│ │ FAILED │ │CONSUMED_IN │ │EXPANDED │
                                             └─────────┘ └──────┘ └────────┘ │   _BATCH   │ └─────────┘
                                                                              └────────────┘

                                             ┌──────────┐
                                             │COALESCED │  ← from coalesce merge (consumed branch tokens)
                                             └──────────┘
```

**Terminal states** (all except BUFFERED): COMPLETED, ROUTED, FORKED, FAILED, QUARANTINED, CONSUMED_IN_BATCH, COALESCED, EXPANDED. Every token reaches exactly one terminal state — no silent drops.

**State origins:**
- `QUARANTINED` — Source validation failure (not from FAILED)
- `COALESCED` — Consumed in a coalesce merge (not from FAILED)
- `EXPANDED` — Parent of a deaggregation 1→N expansion

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.2 | 2026-02-13 | RC-3 alignment — Removed plugin gate references (GateProtocol, BaseGate, PluginGateReason, GateResult, GateOutcome). Gate plugins were removed from the codebase on 2026-02-11. All gates are config-driven via GateSettings + ExpressionParser. |
| 1.1 | 2026-02-08 | Accuracy pass — Fixed RoutingReason union type (added TransformErrorReason, SourceQuarantineReason), corrected terminal states diagram (QUARANTINED/COALESCED/EXPANDED as independent states), fixed FORK_TO_PATHS minimum, added ExpressionSyntaxError, expanded fork_token() signature, added MANUAL trigger type, expanded expression language/forbidden constructs |
| 1.0 | 2026-02-08 | Initial contract — Gates (config + plugin), Forks, Coalesces, Aggregation, Token identity, Routing primitives |
