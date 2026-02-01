# Implementation Plan: AUD-002 Explicit Continue Routing (Revised)

**Status:** ✅ IMPLEMENTED (2026-01-22)

## Implementation Summary

**✅ COMPLETE** - Continue routing decisions now explicitly recorded:
- ✅ Bug P1-2026-01-19-gate-continue-routing-not-recorded.md **CLOSED**
- ✅ `executors.py:414-415` records continue routing with explicit `# AUD-002` comment
- ✅ `executors.py:577-581` records config gate continue with `# AUD-002` comment
- ✅ Continue edges exist in DAG for all gates
- ✅ Route resolution map includes "continue" destinations

**Evidence:** 25 files reference continue routing infrastructure

---

## Problem Statement (Historical)

Gate "continue" decisions are not recorded in the audit trail. When a gate evaluates to "continue" (proceed to next transform), no `RoutingEvent` is created. This means continue decisions are **inferred from absence** rather than explicitly recorded.

### Current Behavior

```
Gate evaluates condition → returns RoutingKind.CONTINUE → no routing event → loop continues
```

The audit trail shows:
- `node_state` with status="completed"
- **ZERO** `routing_events` for that state

### Desired Behavior

```
Gate evaluates condition → returns RoutingKind.CONTINUE → routing event recorded → loop continues
```

## Architecture Review Findings (Addressed)

The architecture review identified these critical issues:

| Issue | Severity | Resolution |
|-------|----------|------------|
| Edge labeling strategy undefined | HIGH | Choose Option A (single "continue" edge) |
| Plugin-based gates not addressed | HIGH | Update both `execute_gate()` AND `execute_config_gate()` |
| RoutingAction type mismatch | HIGH | Use "continue" label consistently |
| Misleading root cause analysis | MEDIUM | Corrected in this revision |
| Missing test cases | MEDIUM | Added to test plan |

## Edge Labeling Strategy Decision: Option A

**Decision:** Create a single edge with `label="continue"` from each gate to its next node.

**Rationale:**
- Simpler implementation - one edge per gate regardless of how many routes resolve to "continue"
- Consistent with transform-to-transform edges which all use "continue" label
- The routing event records which route_label was evaluated; the edge just shows the path taken
- Avoids duplicate edge creation when multiple routes resolve to "continue"

**Trade-off:** The specific route label (e.g., "false") is recorded in `RoutingAction.reason`, not the edge label. This is acceptable because:
1. The `routing_events.reason_hash` can store the evaluation details
2. The edge represents the physical path, not the logical decision

## Solution Design

### Step 1: Create Continue Edges in DAG

**File:** `src/elspeth/core/dag.py`

**Change:** After processing all config gates, create a "continue" edge from each gate to the next node in sequence.

```python
# Track gate sequence during processing
gate_sequence: list[tuple[str, GateSettings]] = []

for gate_config in config.gates:
    gid = node_id("config_gate", gate_config.name)
    # ... existing node creation ...
    gate_sequence.append((gid, gate_config))
    # ... existing route processing (skip continue as before) ...
    prev_node_id = gid

# After gate loop: create continue edges for gates that have continue routes
for i, (gate_id, gate_config) in enumerate(gate_sequence):
    # Check if any route resolves to "continue"
    has_continue_route = any(target == "continue" for target in gate_config.routes.values())

    if has_continue_route:
        # Determine next node
        if i + 1 < len(gate_sequence):
            next_node = gate_sequence[i + 1][0]  # Next gate
        else:
            next_node = output_sink_node  # Output sink

        # Create continue edge (only if not already exists)
        if not graph._graph.has_edge(gate_id, next_node, key="continue"):
            graph.add_edge(gate_id, next_node, label="continue", mode=RoutingMode.MOVE)
```

### Step 2: Record Continue Routing Events in BOTH Executors

**File:** `src/elspeth/engine/executors.py`

#### Change 1: `execute_gate()` (lines 393-395)

```python
if action.kind == RoutingKind.CONTINUE:
    # Record explicit continue routing for audit completeness
    self._record_routing(
        state_id=state.state_id,
        node_id=gate.node_id,
        action=RoutingAction.route("continue", mode=RoutingMode.MOVE),
    )
```

#### Change 2: `execute_gate()` (lines 406-408)

```python
if destination == "continue":
    # Route label resolves to "continue" - record routing event
    self._record_routing(
        state_id=state.state_id,
        node_id=gate.node_id,
        action=RoutingAction.route("continue", mode=RoutingMode.MOVE),
    )
```

#### Change 3: `execute_config_gate()` (lines 565-567)

```python
if destination == "continue":
    action = RoutingAction.continue_(reason=reason)
    # Record explicit continue routing for audit completeness
    self._record_routing(
        state_id=state.state_id,
        node_id=node_id,
        action=RoutingAction.route("continue", mode=RoutingMode.MOVE, reason=reason),
    )
```

### Step 3: Handle Edge Cases

#### Case A: Gate is last in pipeline
- Continue edge points to `output_sink_node`
- This is handled by the "else" branch in Step 1

#### Case B: Gate → Gate continuation
- Continue edge points to next gate in sequence
- This is handled by the `gate_sequence[i + 1]` lookup

#### Case C: Multiple routes resolve to "continue"
- Only ONE continue edge is created (Option A decision)
- All continue routing events use the same edge

#### Case D: No routes resolve to "continue"
- No continue edge is created (gate always routes to sinks)
- `has_continue_route` check handles this

## Implementation Steps

### Phase 1: DAG Changes
1. Add `gate_sequence` tracking in `from_config()`
2. After gate loop, create continue edges for gates with continue routes
3. Add `has_edge` check to prevent duplicates

### Phase 2: Executor Changes
1. Update `execute_gate()` at TWO locations (lines 393-395 and 406-408)
2. Update `execute_config_gate()` at ONE location (lines 565-567)
3. All three use `RoutingAction.route("continue", ...)` for consistency

### Phase 3: Tests

| Test | Description |
|------|-------------|
| `test_dag_creates_continue_edge_for_gate` | Verify continue edge created when route resolves to continue |
| `test_dag_no_continue_edge_when_all_routes_to_sinks` | Verify no edge when no continue routes |
| `test_dag_continue_edge_gate_to_gate` | Verify edge points to next gate (not sink) |
| `test_execute_gate_continue_records_routing` | Plugin gate continue records event |
| `test_execute_gate_route_to_continue_records_routing` | Plugin gate route→continue records event |
| `test_execute_config_gate_continue_records_routing` | Config gate continue records event |

## Files to Modify

| File | Changes |
|------|---------|
| `src/elspeth/core/dag.py` | Create continue edges after gate loop |
| `src/elspeth/engine/executors.py` | Record routing in 3 locations |
| `tests/core/test_dag.py` | Add edge creation tests |
| `tests/engine/test_executors.py` | Add routing event tests |

## Assumptions

1. **DAG structure:** Transforms → Aggregations → Gates → Coalesce → Sink (current order)
2. **No transforms after gates:** Future versions may need adjustment if this changes
3. **Backwards compatibility:** Runs before this change will have different audit data (no continue events)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| MissingEdgeError at runtime | LOW | Pipeline crashes | Option A ensures edge exists |
| Duplicate edge creation | LOW | Graph validation fails | `has_edge` check |
| Performance impact | LOW | Minimal | One extra edge per gate, one DB write per continue |

## Estimated Effort

- DAG changes: 45 minutes
- Executor changes: 30 minutes
- Tests: 1 hour
- **Total: ~2.5 hours**
