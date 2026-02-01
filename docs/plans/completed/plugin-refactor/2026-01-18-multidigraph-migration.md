# MultiDiGraph Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate ExecutionGraph from NetworkX DiGraph to MultiDiGraph to support multiple labeled edges between the same node pair (required for fork operations).

**Architecture:** Replace the underlying `DiGraph` with `MultiDiGraph` in `ExecutionGraph`, using edge labels as keys. This enables fork gates to create multiple edges to the same destination (e.g., `path_a` and `path_b` both going to the output sink). The change is encapsulated within `ExecutionGraph` - external code uses the same public API.

**Tech Stack:** NetworkX MultiDiGraph, existing EdgeInfo contract, pytest for testing.

**Bug Reference:** Closes `docs/bugs/pending/2026-01-15-gate-multiple-labels-same-sink.md`

---

## Background

### The Problem

`ExecutionGraph` wraps NetworkX's `DiGraph`, which only allows ONE edge between any two nodes. When a fork gate creates children going to the same destination:

```
ForkGate ──path_a──► Sink
         ──path_b──► Sink  (OVERWRITES path_a!)
```

Only `path_b` survives. This breaks:
- Fork operations where children converge to same sink
- Gates with multiple route labels to the same sink (e.g., `{"high": "alerts", "medium": "alerts"}`)

### The Solution

`MultiDiGraph` allows multiple edges between nodes, distinguished by a key:

```python
# DiGraph: graph[u][v] = edge_data
# MultiDiGraph: graph[u][v] = {key1: edge_data1, key2: edge_data2, ...}
```

We use the edge `label` as the key, so both `path_a` and `path_b` coexist.

---

## Task 1: Update ExecutionGraph to Use MultiDiGraph

**Files:**
- Modify: `src/elspeth/core/dag.py:16-17, 46-47, 90-106, 206-220, 380`
- Modify: `tests/engine/test_engine_gates.py:140` (test helper also uses `has_edge`)

### Step 1: Write failing test for multi-edge support

```python
# tests/core/test_dag.py - add to TestDAGBuilder class

def test_multiple_edges_same_node_pair(self) -> None:
    """MultiDiGraph allows multiple labeled edges between same nodes."""
    from elspeth.contracts import RoutingMode
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()
    graph.add_node("gate", node_type="gate", plugin_name="fork_gate")
    graph.add_node("sink", node_type="sink", plugin_name="output")

    # Add two edges with different labels to SAME destination
    graph.add_edge("gate", "sink", label="path_a", mode=RoutingMode.COPY)
    graph.add_edge("gate", "sink", label="path_b", mode=RoutingMode.COPY)

    # Both edges should exist (DiGraph would show 1, MultiDiGraph shows 2)
    assert graph.edge_count == 2

    edges = graph.get_edges()
    labels = {e.label for e in edges}
    assert labels == {"path_a", "path_b"}

def test_multi_edge_graph_is_acyclic(self) -> None:
    """Verify is_acyclic() works correctly with MultiDiGraph parallel edges."""
    from elspeth.contracts import RoutingMode
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv")
    graph.add_node("gate", node_type="gate", plugin_name="classifier")
    graph.add_node("sink", node_type="sink", plugin_name="csv")

    graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
    # Multiple parallel edges to same sink - still acyclic
    graph.add_edge("gate", "sink", label="high", mode=RoutingMode.MOVE)
    graph.add_edge("gate", "sink", label="medium", mode=RoutingMode.MOVE)
    graph.add_edge("gate", "sink", label="low", mode=RoutingMode.MOVE)

    # Graph with parallel edges should still be detected as acyclic
    assert graph.is_acyclic() is True
    # Full validation should also pass
    graph.validate()
```

### Step 2: Run tests to verify they fail

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestDAGBuilder::test_multiple_edges_same_node_pair tests/core/test_dag.py::TestDAGBuilder::test_multi_edge_graph_is_acyclic -v
```

Expected: FAIL - `assert graph.edge_count == 2` fails (actual: 1) because DiGraph overwrites edges

### Step 3: Update import and type annotation

Change the import and graph type in `dag.py`:

```python
# Line 16: Change import
# BEFORE:
from networkx import DiGraph

# AFTER:
from networkx import MultiDiGraph

# Line 47: Change type annotation
# BEFORE:
self._graph: DiGraph[str] = nx.DiGraph()

# AFTER:
self._graph: MultiDiGraph = nx.MultiDiGraph()
```

### Step 4: Update add_edge() to use label as key

```python
# Line 90-106: Update add_edge method
def add_edge(
    self,
    from_node: str,
    to_node: str,
    *,
    label: str,
    mode: RoutingMode = RoutingMode.MOVE,
) -> None:
    """Add an edge between nodes.

    Args:
        from_node: Source node ID
        to_node: Target node ID
        label: Edge label (e.g., "continue", "suspicious") - also used as edge key
        mode: Routing mode (MOVE or COPY)
    """
    # Use label as key to allow multiple edges between same nodes
    self._graph.add_edge(from_node, to_node, key=label, label=label, mode=mode)
```

### Step 5: Update get_edges() for MultiDiGraph iteration

```python
# Line 206-220: Update get_edges method
def get_edges(self) -> list[EdgeInfo]:
    """Get all edges with their data as typed EdgeInfo.

    Returns:
        List of EdgeInfo contracts (not tuples)
    """
    # Note: _key is unused but required for MultiDiGraph iteration signature
    return [
        EdgeInfo(
            from_node=u,
            to_node=v,
            label=data["label"],
            mode=data["mode"],
        )
        for u, v, _key, data in self._graph.edges(data=True, keys=True)
    ]
```

> **Note:** Use `_key` (with underscore) to satisfy ruff's unused-variable check. The key is required in the iteration signature but we already have `label` in `data["label"]`.

### Step 6: Update has_edge() check in from_config()

```python
# Line 380: Update has_edge check to use specific key
# BEFORE:
if not graph._graph.has_edge(prev_node_id, output_sink_node):

# AFTER:
if not graph._graph.has_edge(prev_node_id, output_sink_node, key="continue"):
```

### Step 6.5: Update has_edge() check in test helper

The test helper in `tests/engine/test_engine_gates.py` also builds graphs manually and has the same pattern:

```python
# Line 140: Update has_edge check to use specific key
# BEFORE:
if not graph._graph.has_edge(prev, output_sink_node):

# AFTER:
if not graph._graph.has_edge(prev, output_sink_node, key="continue"):
```

### Step 7: Run tests to verify they pass

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestDAGBuilder::test_multiple_edges_same_node_pair tests/core/test_dag.py::TestDAGBuilder::test_multi_edge_graph_is_acyclic -v
```

Expected: Both tests PASS

### Step 8: Run all DAG tests to verify no regressions

```bash
.venv/bin/python -m pytest tests/core/test_dag.py -v
```

Expected: All tests PASS

### Step 9: Commit

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py tests/engine/test_engine_gates.py
git commit -m "$(cat <<'EOF'
feat(dag): migrate ExecutionGraph from DiGraph to MultiDiGraph

- Replace networkx.DiGraph with MultiDiGraph to support multiple edges
  between the same node pair (required for fork operations)
- Use edge label as key in add_edge() for edge uniqueness
- Update get_edges() iteration for MultiDiGraph (includes keys=True)
- Update has_edge() checks to use specific key="continue"
- Add test verifying multiple edges to same destination
- Add test verifying is_acyclic() works with parallel edges

Closes: docs/bugs/pending/2026-01-15-gate-multiple-labels-same-sink.md

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Run Full Test Suite

**Files:**
- None (verification only)

### Step 1: Run all engine tests

```bash
.venv/bin/python -m pytest tests/engine/ -v
```

Expected: All tests PASS

### Step 1.5: Run gates tests specifically

The gates tests use the modified test helper, so verify they still pass:

```bash
.venv/bin/python -m pytest tests/engine/test_engine_gates.py -v
```

Expected: All tests PASS

### Step 2: Run all core tests

```bash
.venv/bin/python -m pytest tests/core/ -v
```

Expected: All tests PASS

### Step 3: Run mypy type check

```bash
.venv/bin/python -m mypy src/elspeth/core/dag.py --strict
```

Expected: No errors

---

## Task 3: Add Integration Tests for Multi-Edge Scenarios

**Files:**
- Modify: `tests/core/test_dag.py`

> **Note:** The second test (`test_gate_multiple_routes_same_sink`) directly exercises the bug scenario from the bug report - multiple route labels (high, medium, low) all routing to the same sink. The first test validates that fork configurations parse correctly but doesn't exercise the multi-edge bug because `"fork"` routes don't create edges to sinks.

### Step 1: Write tests for graph construction scenarios

```python
# tests/core/test_dag.py - add new test class

class TestMultiEdgeScenarios:
    """Tests for scenarios requiring multiple edges between same nodes."""

    def test_fork_gate_config_parses_into_valid_graph(self) -> None:
        """Fork gate configuration parses into valid graph structure.

        Note: This tests config parsing, not the multi-edge bug. Fork routes
        with target="fork" don't create edges to sinks - they create child tokens.
        The multi-edge bug is tested by test_gate_multiple_routes_same_sink.
        """
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"output": SinkSettings(plugin="csv")},
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="true",  # Always forks
                    routes={"path_a": "fork", "path_b": "fork"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            output_sink="output",
        )

        graph = ExecutionGraph.from_config(config)

        # Validate graph is still valid (DAG, has source and sink)
        graph.validate()

        # The gate should have edges - at minimum the continue edge to output sink
        edges = graph.get_edges()
        gate_edges = [e for e in edges if "config_gate" in e.from_node]

        # Should have at least the continue edge to output sink
        assert len(gate_edges) >= 1

    def test_gate_multiple_routes_same_sink(self) -> None:
        """CRITICAL: Gate with multiple route labels to same sink preserves all labels.

        This is the core bug scenario: {"high": "alerts", "medium": "alerts", "low": "alerts"}
        With DiGraph, only "low" survives. With MultiDiGraph, all three edges exist.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("gate", node_type="gate", plugin_name="classifier")
        graph.add_node("alerts", node_type="sink", plugin_name="csv")

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        # Multiple severity levels all route to same alerts sink
        graph.add_edge("gate", "alerts", label="high", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "alerts", label="medium", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "alerts", label="low", mode=RoutingMode.MOVE)

        # All three edges should exist
        edges = graph.get_edges()
        alert_edges = [e for e in edges if e.to_node == "alerts"]
        assert len(alert_edges) == 3

        labels = {e.label for e in alert_edges}
        assert labels == {"high", "medium", "low"}
```

### Step 2: Run new tests

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestMultiEdgeScenarios -v
```

Expected: All tests PASS (especially `test_gate_multiple_routes_same_sink`)

### Step 3: Commit

```bash
git add tests/core/test_dag.py
git commit -m "$(cat <<'EOF'
test(dag): add multi-edge scenario integration tests

- Test fork gate config parsing creates valid graph structure
- Test multiple route labels to same sink all preserved (core bug fix)
- Verifies MultiDiGraph migration works for real use cases

The test_gate_multiple_routes_same_sink test directly exercises the
bug scenario: {"high": "alerts", "medium": "alerts", "low": "alerts"}

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update Bug Report Status

**Files:**
- Move: `docs/bugs/pending/2026-01-15-gate-multiple-labels-same-sink.md` → `docs/bugs/closed/`

### Step 1: Move bug report to closed

```bash
mkdir -p docs/bugs/closed
git mv docs/bugs/pending/2026-01-15-gate-multiple-labels-same-sink.md docs/bugs/closed/
```

### Step 2: Add resolution note to bug report

Add to the end of the file:

```markdown
## Resolution

**Fixed in:** 2026-01-18
**Fix:** Migrated `ExecutionGraph` from `networkx.DiGraph` to `networkx.MultiDiGraph`, using edge labels as keys to allow multiple edges between the same node pair.

**Commits:**
- feat(dag): migrate ExecutionGraph from DiGraph to MultiDiGraph
- test(dag): add fork and multi-route graph construction tests
```

### Step 3: Commit

```bash
git add docs/bugs/
git commit -m "$(cat <<'EOF'
docs(bugs): close gate-multiple-labels-same-sink bug

Bug fixed by MultiDiGraph migration - multiple route labels to same
sink now supported.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final Verification

**Files:**
- None (verification only)

### Step 1: Run complete test suite

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

Expected: All tests PASS

### Step 2: Run type checking on modified files

```bash
.venv/bin/python -m mypy src/elspeth/core/dag.py src/elspeth/engine/orchestrator.py --strict
```

Expected: No errors

### Step 3: Run linter

```bash
.venv/bin/python -m ruff check src/elspeth/core/dag.py
```

Expected: No errors

---

## Verification Checklist

After all tasks complete, verify:

- [ ] `ExecutionGraph` uses `MultiDiGraph` internally
- [ ] `add_edge()` uses label as edge key
- [ ] `get_edges()` iterates with `keys=True` and uses `_key` (underscore prefix)
- [ ] `has_edge()` check in `from_config()` uses specific key
- [ ] `has_edge()` check in test helper uses specific key
- [ ] Multiple edges between same nodes are preserved
- [ ] `is_acyclic()` works correctly with parallel edges
- [ ] `edge_count` returns correct count including parallel edges
- [ ] All existing tests pass (no regressions)
- [ ] New multi-edge test passes
- [ ] New is_acyclic() test passes
- [ ] Fork config parsing test passes
- [ ] Multi-route to same sink test passes (CRITICAL - exercises the bug)
- [ ] Bug report moved to closed
- [ ] mypy passes on dag.py
- [ ] ruff passes on dag.py (no unused variable warnings)

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Breaking existing edge iteration | Low | `get_edges()` returns same `EdgeInfo` type |
| Performance regression | Very Low | MultiDiGraph has same O(1) edge operations |
| Subtle behavior change in validation | Low | NetworkX's `is_acyclic()` works on MultiDiGraph |
| Test helper accessing `_graph` directly | Low | `tests/engine/test_engine_gates.py:140` updated in Step 6.5 |

---

## Notes

1. **Encapsulation wins:** Because `ExecutionGraph` properly encapsulates the NetworkX graph, this change only touches `dag.py`. External code (`orchestrator.py`, `processor.py`, `executors.py`) continues working unchanged.

2. **Edge key strategy:** Using `label` as the edge key is natural because labels are already unique per routing decision. Two edges can go to the same sink if they have different labels (`path_a`, `path_b`).

3. **No migration needed:** This is a pure implementation change. No data migration, no config changes, no API changes.

4. **Behavioral change (correct):** The `edge_count` property will now return the correct count of ALL edges including parallel edges. Previously, parallel edges overwrote each other, so `edge_count` was artificially low. This is the fix, not a regression.

5. **Unused variable convention:** Use `_key` (underscore prefix) in `get_edges()` iteration. The key is required by the MultiDiGraph iteration signature but we already have the label in `data["label"]`. The underscore tells ruff/pylint it's intentionally unused.
