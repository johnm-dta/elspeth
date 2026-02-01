# BUG: Coalesce Graph Topology Does Not Match Processor Execution

**Severity:** P2 (Design Flaw - Correctness/Auditability Impact)
**Component:** Engine (Processor, DAG)
**Discovered:** 2026-01-27
**Status:** RESOLVED (2026-01-28)

---

## Resolution Summary

**Implemented Option A: Processor now matches graph topology.**

Fork children now skip directly to their coalesce step, bypassing all intermediate
transforms and gates. Merged tokens can continue to downstream nodes if the coalesce
is not at the end of the pipeline.

### Key Changes

1. **`src/elspeth/core/dag.py`**
   - Added `_coalesce_gate_index` instance variable to `ExecutionGraph`
   - Added `get_coalesce_gate_index()` getter method

2. **`src/elspeth/engine/orchestrator.py`**
   - Added `_compute_coalesce_step_map()` helper method
   - Formula: `coalesce_step = num_transforms + num_gates + coalesce_index`
   - This places coalesce steps AFTER all transforms/gates, avoiding step index collisions

3. **`src/elspeth/engine/processor.py`**
   - Updated plugin gate fork path (~line 905): children get `start_step=coalesce_step`
   - Updated config gate fork path (~line 1139): children get `start_step=coalesce_step`

### Test Coverage

- `test_multiple_gates_fork_coalesce_step_index`: Verifies no step index collision
- `test_fork_coalesce_merged_token_has_terminal_outcome`: Verifies audit completeness
- `test_fork_coalesce_all_tokens_have_correct_outcomes`: Verifies token outcomes
- Full test suite: 3887 passed, 33 skipped

---

## Summary

The DAG graph wires fork branches directly to coalesce nodes, with merged tokens continuing to downstream nodes. However, the processor executes fork branches through ALL remaining transforms/gates before coalescing, and merged tokens go directly to sink. This mismatch means the graph visualization does not accurately represent execution, which is problematic for an auditable system.

---

## Current Behavior

### Graph Topology (dag.py:626-631, 728-737)

```
fork_gate --branch_a--> coalesce --continue--> downstream_gate --continue--> sink
fork_gate --branch_b--> coalesce
```

The graph construction:
1. Fork branches are wired **directly to coalesce** (dag.py:626-631)
2. Coalesce continues to **next pipeline node** after the fork gate (dag.py:732-737)
3. `coalesce_gate_index` tracks which gate produces each coalesce's branches

### Processor Execution (processor.py:1132-1153, 772-789)

```
fork_gate --> branch_a --> downstream_gate --> coalesce --> sink
         --> branch_b --> downstream_gate --> coalesce
```

The processor execution:
1. Fork children get `start_step = len(transforms) + next_config_step` (line 1149)
2. Children process **all remaining gates** before coalescing
3. Coalesce triggers only when `step_completed >= coalesce_at_step`
4. Merged token returns `COALESCED` immediately (never continues)

---

## Impact

### 1. Audit Trail Mismatch
The graph shown to auditors does not match actual execution. If an auditor asks "what path did this row take?", the graph suggests one flow but execution did another.

### 2. Unexpected Duplicate Processing
For a pipeline `source → fork_gate → transform2 → coalesce → sink`:
- **Expected** (from graph): transform2 runs once on merged result
- **Actual**: transform2 runs twice, once per branch

### 3. Dead Code Path
The "merged token continues" code (processor.py:783-789) checks `if coalesce_at_step < total_steps`, but:
- Before P1 fix: `coalesce_at_step = total_steps` → condition always False
- After P1 fix: `coalesce_at_step = total_steps + 1` → condition still False

This path is never executed, suggesting the feature is incomplete.

---

## Reproduction

```python
# Pipeline with gate AFTER fork
settings = ElspethSettings(
    source={"plugin": "null"},
    gates=[
        GateSettings(
            name="fork_gate",
            condition="True",
            routes={"true": "fork", "false": "continue"},
            fork_to=["branch_a", "branch_b"],
        ),
        GateSettings(
            name="passthrough_gate",  # Should run on merged token per graph
            condition="False",
            routes={"true": "discard", "false": "continue"},
        ),
    ],
    sinks={"output": SinkSettings(plugin="json", options={...})},
    coalesce=[
        CoalesceSettings(
            name="merge_branches",
            branches=["branch_a", "branch_b"],
            policy="require_all",
            merge="union",
        ),
    ],
    default_sink="output",
)

# Graph topology:
#   fork_gate --branch_a--> coalesce --continue--> passthrough_gate --> sink
#   fork_gate --branch_b--> coalesce
#
# Actual execution:
#   fork_gate --> branch_a --> passthrough_gate --> coalesce --> sink
#             --> branch_b --> passthrough_gate --> coalesce
#
# Result: passthrough_gate runs TWICE (per branch), not ONCE (on merged token)
```

---

## Root Cause Analysis

### Graph Construction (dag.py)

```python
# Line 626-631: Fork branches wired directly to coalesce
for branch_name in gate_entry.fork_to:
    if BranchName(branch_name) in branch_to_coalesce:
        coalesce_name = branch_to_coalesce[BranchName(branch_name)]
        coalesce_id = coalesce_ids[coalesce_name]
        graph.add_edge(gate_entry.node_id, coalesce_id, label=branch_name, mode=RoutingMode.COPY)

# Line 732-737: Coalesce continues to next node
gate_idx = coalesce_gate_index[coalesce_name]
if gate_idx + 1 < len(pipeline_nodes):
    next_node_id = pipeline_nodes[gate_idx + 1]
else:
    next_node_id = sink_ids[SinkName(default_sink)]
graph.add_edge(coalesce_id, next_node_id, label="continue", mode=RoutingMode.MOVE)
```

### Processor Execution (processor.py)

```python
# Line 1149: Fork children start at NEXT gate, not at coalesce
child_items.append(
    _WorkItem(
        token=child_token,
        start_step=len(transforms) + next_config_step,  # Next gate, not coalesce!
        coalesce_at_step=cfg_coalesce_step,
        coalesce_name=cfg_coalesce_name,
    )
)

# Line 773-781: Merged token always returns COALESCED (never continues)
if coalesce_at_step >= total_steps:  # Always true with current calculation
    return (
        True,
        RowResult(
            token=coalesce_outcome.merged_token,
            final_data=coalesce_outcome.merged_token.row_data,
            outcome=RowOutcome.COALESCED,
        ),
    )
```

---

## Proposed Fix

### Option A: Make Processor Match Graph (Recommended)

Change processor to respect graph topology:

1. **Fork children skip to coalesce directly**
   ```python
   # Instead of: start_step = len(transforms) + next_config_step
   # Use: start_step = coalesce_at_step (children skip intermediate nodes)
   ```

2. **Derive coalesce_at_step from graph insertion point**
   ```python
   # Compute from coalesce_gate_index, not hardcoded to end
   coalesce_at_step = coalesce_gate_index[coalesce_name] + 1
   ```

3. **Enable merged token continuation**
   ```python
   # When coalesce_at_step < total_steps, merged token continues
   # This path exists but is never reached with current calculation
   ```

**Pros:**
- Graph accurately represents execution
- Auditors see correct flow
- More intuitive pipeline semantics

**Cons:**
- Breaking change for existing pipelines
- Requires careful migration

### Option B: Make Graph Match Processor

Change graph construction to reflect current execution:

1. Wire fork branches through remaining gates to coalesce
2. Wire coalesce directly to sink (no continuation)

**Pros:**
- No execution changes
- Existing pipelines unaffected

**Cons:**
- Graph becomes complex (shows branch duplication)
- Less intuitive model
- "Merged token continues" feature unusable

### Option C: Support Both Models (Future)

Add explicit configuration for coalesce behavior:
- `coalesce_point: "immediate"` - branches skip to coalesce (graph model)
- `coalesce_point: "end_of_pipeline"` - branches process all remaining nodes (current)

---

## Recommended Approach

**Implement Option A** with the following steps:

1. **Add feature flag** for gradual rollout
2. **Update processor** to derive coalesce step from graph
3. **Update fork child routing** to skip to coalesce
4. **Enable merged token continuation** path
5. **Add comprehensive tests** for the corrected behavior
6. **Update documentation** to clarify coalesce semantics

---

## Files Affected

| File | Changes Required |
|------|------------------|
| `src/elspeth/engine/processor.py` | Fork child start_step, merged token continuation |
| `src/elspeth/engine/orchestrator.py` | coalesce_step_map derivation from graph |
| `src/elspeth/core/dag.py` | Expose coalesce_gate_index via method |
| `tests/engine/test_*.py` | New tests for graph-aligned execution |

---

## Related Issues

- **P1 Fix (Commit 5bc49a6):** Step index collision fix uses `base_step + 1 + i`, which exacerbates the "merged token never continues" issue by making `coalesce_at_step > total_steps` instead of `==`.

---

## Test Cases for Fix

1. **Fork → Coalesce → Transform → Sink**: Verify transform runs on merged token only
2. **Fork → Coalesce → Gate → Sink**: Verify gate evaluates merged token only
3. **Fork → Coalesce → Fork → Coalesce → Sink**: Nested coalesce with continuation
4. **Audit trail verification**: Graph matches actual node traversals
5. **Backward compatibility**: Existing fork-at-end-of-pipeline still works
