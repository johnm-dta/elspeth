# Schema Inheritance for Gate Nodes Implementation Plan (v2 - Corrected)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable schema validation for gate-routed edges by implementing schema inheritance for pass-through nodes.

**Architecture:** Gates are routing nodes that pass data unchanged. Currently they lack `input_schema`/`output_schema`, causing validation to skip gate edges. We'll add a helper method to walk backwards through the graph to find the effective producer schema, with proper multi-input validation, then use this when validating edges involving gates.

**Tech Stack:** Python 3.12+, NetworkX (MultiDiGraph), Pydantic (PluginSchema), pytest

---

## Background Context

**Current State:**
- `ExecutionGraph._validate_edge_schemas()` in `src/elspeth/core/dag.py` validates schema compatibility along edges
- Validation skips edges where either node has `schema = None`
- Gates, aggregations, and coalesce nodes are added without schemas in the `from_config()` method

**Bug Impact:**
- Config-driven gates (built via `from_config()`) lack schemas
- Gate routes to sinks mid-pipeline bypass schema validation
- Incompatible data flows through gates to sinks, causing runtime failures
- Violates ELSPETH auditability standard: "if it's not recorded, it didn't happen"

**Fix Scope:**
This plan addresses **gates only**. Aggregations and coalesce nodes require plugin-level schema support (separate work).

**Important Note on Existing Test:**
The test `test_schema_validation_catches_gate_routing_to_incompatible_sink` MANUALLY assigns `input_schema` and `output_schema` to the gate node, so it validates a different scenario (manually-constructed gates with schemas). Our fix targets config-driven gates that have NO schemas.

---

## Task 1: Add `get_incoming_edges()` Helper Method

**Files:**
- Modify: `src/elspeth/core/dag.py` (add method after the `get_edges()` method)
- Test: `tests/core/test_dag.py` (add test in existing `TestExecutionGraph` class)

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py` at end of `TestExecutionGraph` class:

```python
def test_get_incoming_edges_returns_edges_pointing_to_node(self):
    """get_incoming_edges() returns all edges with to_node matching the given node_id."""
    from elspeth.core.dag import ExecutionGraph, EdgeInfo
    from elspeth.contracts import RoutingMode

    graph = ExecutionGraph()
    graph.add_node("A", node_type="source", plugin_name="csv")
    graph.add_node("B", node_type="transform", plugin_name="mapper")
    graph.add_node("C", node_type="sink", plugin_name="csv")

    graph.add_edge("A", "B", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("B", "C", label="continue", mode=RoutingMode.MOVE)

    incoming = graph.get_incoming_edges("B")

    assert len(incoming) == 1
    assert incoming[0].from_node == "A"
    assert incoming[0].to_node == "B"
    assert incoming[0].label == "continue"
    assert incoming[0].mode == RoutingMode.MOVE
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_get_incoming_edges_returns_edges_pointing_to_node -v
```

Expected: FAIL with `AttributeError: 'ExecutionGraph' object has no attribute 'get_incoming_edges'`

**Step 3: Write minimal implementation**

Add method to `src/elspeth/core/dag.py` after the `get_edges()` method:

```python
def get_incoming_edges(self, node_id: str) -> list[EdgeInfo]:
    """Get all edges pointing TO this node.

    Args:
        node_id: The target node ID

    Returns:
        List of EdgeInfo for edges where to_node == node_id
    """
    # NetworkX in_edges returns (from, to, key) tuples for MultiDiGraph
    return [
        EdgeInfo(
            from_node=u,
            to_node=v,
            label=data["label"],
            mode=data["mode"],
        )
        for u, v, _key, data in self._graph.in_edges(node_id, data=True, keys=True)
    ]
```

**Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_get_incoming_edges_returns_edges_pointing_to_node -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/core/test_dag.py src/elspeth/core/dag.py
git commit -m "feat(dag): add get_incoming_edges() method for schema inheritance

Add helper method to retrieve all edges pointing to a node. Required for
walking backwards through pass-through nodes (gates) to find effective
producer schema.

Refs: P1-2026-01-21-schema-validator-ignores-dag-routing"
```

---

## Task 2: Add Test for Empty Incoming Edges Case

**Files:**
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add immediately after previous test:

```python
def test_get_incoming_edges_returns_empty_for_source_node(self):
    """get_incoming_edges() returns empty list for nodes with no predecessors."""
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()
    graph.add_node("A", node_type="source", plugin_name="csv")
    graph.add_node("B", node_type="sink", plugin_name="csv")

    incoming = graph.get_incoming_edges("A")

    assert incoming == []
```

**Step 2: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_get_incoming_edges_returns_empty_for_source_node -v
```

Expected: PASS (implementation already handles this via NetworkX)

**Step 3: Commit**

```bash
git add tests/core/test_dag.py
git commit -m "test(dag): verify get_incoming_edges() handles source nodes correctly"
```

---

## Task 3: Add Helper to Find Effective Producer Schema

**Files:**
- Modify: `src/elspeth/core/dag.py` (add private method after the `_validate_edge_schemas()` method)
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py` in `TestExecutionGraph` class:

```python
def test_get_effective_producer_schema_walks_through_gates(self):
    """_get_effective_producer_schema() recursively finds schema through gate chain."""
    from elspeth.core.dag import ExecutionGraph
    from elspeth.contracts import PluginSchema, RoutingMode

    class OutputSchema(PluginSchema):
        value: int

    graph = ExecutionGraph()

    # Build chain: source -> gate -> sink
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=OutputSchema)
    graph.add_node("gate", node_type="gate", plugin_name="config_gate:check")  # No schema
    graph.add_node("sink", node_type="sink", plugin_name="csv")

    graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("gate", "sink", label="flagged", mode=RoutingMode.MOVE)

    # Gate's effective producer schema should be source's output schema
    effective_schema = graph._get_effective_producer_schema("gate")

    assert effective_schema == OutputSchema
```

**Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_get_effective_producer_schema_walks_through_gates -v
```

Expected: FAIL with `AttributeError: 'ExecutionGraph' object has no attribute '_get_effective_producer_schema'`

**Step 3: Write minimal implementation**

Add private method to `src/elspeth/core/dag.py` after the `_validate_edge_schemas()` method:

```python
def _get_effective_producer_schema(self, node_id: str) -> type[PluginSchema] | None:
    """Get effective output schema for a node, walking through pass-through nodes.

    Gates and other pass-through nodes don't transform data - they inherit
    schema from their upstream producers. This method walks backwards through
    the graph to find the nearest schema-carrying producer.

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

    # Not a pass-through type and no schema - return None
    return None
```

**Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_get_effective_producer_schema_walks_through_gates -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/core/test_dag.py src/elspeth/core/dag.py
git commit -m "feat(dag): add _get_effective_producer_schema() for pass-through nodes

Implement recursive schema resolution that walks backwards through gate
nodes to find the actual data-transforming producer. Gates pass data
unchanged, so their effective output schema is their upstream producer's
output schema.

Validates multi-input gates have compatible schemas (crashes on mismatch).
Crashes if gate has no incoming edges (graph construction bug).

Refs: P1-2026-01-21-schema-validator-ignores-dag-routing"
```

---

## Task 4: Add Test for Gate with No Incoming Edges (Crashes)

**Files:**
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py`:

```python
def test_get_effective_producer_schema_crashes_on_gate_without_inputs(self):
    """_get_effective_producer_schema() crashes if gate has no incoming edges."""
    from elspeth.core.dag import ExecutionGraph, GraphValidationError

    graph = ExecutionGraph()
    graph.add_node("gate", node_type="gate", plugin_name="config_gate:orphan")

    # Gate with no inputs is a bug in our code - should crash
    with pytest.raises(GraphValidationError) as exc_info:
        graph._get_effective_producer_schema("gate")

    assert "no incoming edges" in str(exc_info.value).lower()
    assert "graph construction bug" in str(exc_info.value).lower()
```

**Step 2: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_get_effective_producer_schema_crashes_on_gate_without_inputs -v
```

Expected: PASS (implementation already crashes on this)

**Step 3: Commit**

```bash
git add tests/core/test_dag.py
git commit -m "test(dag): verify schema inheritance crashes on orphan gates

Per CLAUDE.md: crash on our bugs. Gate with no incoming edges is a graph
construction bug, not user data issue."
```

---

## Task 5: Add Test for Chained Gates

**Files:**
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py`:

```python
def test_get_effective_producer_schema_handles_chained_gates(self):
    """_get_effective_producer_schema() recursively walks through multiple gates."""
    from elspeth.core.dag import ExecutionGraph
    from elspeth.contracts import PluginSchema, RoutingMode

    class SourceOutput(PluginSchema):
        id: int
        name: str

    graph = ExecutionGraph()

    # Build chain: source -> gate1 -> gate2 -> sink
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=SourceOutput)
    graph.add_node("gate1", node_type="gate", plugin_name="config_gate:first")
    graph.add_node("gate2", node_type="gate", plugin_name="config_gate:second")
    graph.add_node("sink", node_type="sink", plugin_name="csv")

    graph.add_edge("source", "gate1", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("gate1", "gate2", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("gate2", "sink", label="approved", mode=RoutingMode.MOVE)

    # gate2's effective schema should trace back to source
    effective_schema = graph._get_effective_producer_schema("gate2")

    assert effective_schema == SourceOutput
```

**Step 2: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_get_effective_producer_schema_handles_chained_gates -v
```

Expected: PASS (recursive implementation already handles this)

**Step 3: Commit**

```bash
git add tests/core/test_dag.py
git commit -m "test(dag): verify schema inheritance through chained gates"
```

---

## Task 6: Add Test for Non-Gate Node Returns Direct Schema

**Files:**
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py`:

```python
def test_get_effective_producer_schema_returns_direct_schema_for_transform(self):
    """_get_effective_producer_schema() returns output_schema directly for transform nodes."""
    from elspeth.core.dag import ExecutionGraph
    from elspeth.contracts import PluginSchema

    class TransformOutput(PluginSchema):
        result: str

    graph = ExecutionGraph()
    graph.add_node(
        "transform",
        node_type="transform",
        plugin_name="field_mapper",
        output_schema=TransformOutput
    )

    effective_schema = graph._get_effective_producer_schema("transform")

    assert effective_schema == TransformOutput
```

**Step 2: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_get_effective_producer_schema_returns_direct_schema_for_transform -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/core/test_dag.py
git commit -m "test(dag): verify schema inheritance for non-gate nodes"
```

---

## Task 7: Update `_validate_edge_schemas()` to Use Effective Schema

**Files:**
- Modify: `src/elspeth/core/dag.py` (update the `_validate_edge_schemas()` method)
- Test: `tests/core/test_dag.py`

**Step 1: Understand existing test limitation**

The test `test_schema_validation_catches_gate_routing_to_incompatible_sink` manually assigns schemas to the gate:

```python
# This test MANUALLY sets gate schemas - doesn't test the actual bug
graph.add_node("gate", ..., input_schema=X, output_schema=X)
```

This test validates a DIFFERENT scenario (manually-constructed gates WITH schemas).

Our fix targets config-driven gates that have NO schemas when added by `from_config()`.

**Step 2: Write new test for config-driven gate validation**

Add to `tests/core/test_dag.py`:

```python
def test_validate_edge_schemas_uses_effective_schema_for_gates(self):
    """_validate_edge_schemas() uses effective producer schema for gate edges."""
    from elspeth.core.dag import ExecutionGraph, GraphValidationError
    from elspeth.contracts import PluginSchema, RoutingMode

    class SourceOutput(PluginSchema):
        id: int
        name: str
        # Note: does NOT have 'score' field

    class SinkInput(PluginSchema):
        id: int
        score: float  # Required field not in source output

    graph = ExecutionGraph()

    # Pipeline: source -> gate -> sink
    # Gate has NO schemas (simulates config-driven gate from from_config())
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=SourceOutput)
    graph.add_node("gate", node_type="gate", plugin_name="config_gate:check")  # NO SCHEMA
    graph.add_node("sink", node_type="sink", plugin_name="csv", input_schema=SinkInput)

    graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("gate", "sink", label="flagged", mode=RoutingMode.MOVE)

    # Should detect schema incompatibility on gate -> sink edge
    with pytest.raises(GraphValidationError) as exc_info:
        graph.validate()

    # Verify error mentions the missing field
    assert "score" in str(exc_info.value).lower()
    # Verify error includes plugin names (config_gate:check -> csv)
    assert "config_gate:check" in str(exc_info.value)
    assert "csv" in str(exc_info.value)
```

**Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_validate_edge_schemas_uses_effective_schema_for_gates -v
```

Expected: FAIL - validation doesn't raise error because gate edges are currently skipped

**Step 4: Update `_validate_edge_schemas()` implementation**

Modify the `_validate_edge_schemas()` method in `src/elspeth/core/dag.py`. Replace the method with:

```python
def _validate_edge_schemas(self) -> list[str]:
    """Validate schema compatibility along all edges.

    For each edge (producer -> consumer):
    - Get producer's effective output schema (walks through gates)
    - Get consumer's input schema
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

        # Get consumer input schema directly
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

**Step 5: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_validate_edge_schemas_uses_effective_schema_for_gates -v
```

Expected: PASS

**Step 6: Run all DAG tests to ensure no regressions**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py -v
```

Expected: All tests PASS

**Step 7: Commit**

```bash
git add tests/core/test_dag.py src/elspeth/core/dag.py
git commit -m "fix(dag): use effective producer schema in edge validation

Update _validate_edge_schemas() to call _get_effective_producer_schema()
instead of accessing output_schema directly. This enables validation of
edges involving gates, which pass data unchanged from their upstream
producers.

Fixes: P1-2026-01-21-schema-validator-ignores-dag-routing (gate routing)

BREAKING CHANGE: Schema validation now catches gate routing to
incompatible sinks that previously passed validation silently."
```

---

## Task 8: Add Test for Fork Gate Validation

**Files:**
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py`:

```python
def test_validate_edge_schemas_validates_all_fork_destinations(self):
    """Fork gates validate all destination edges against effective schema."""
    from elspeth.core.dag import ExecutionGraph, GraphValidationError
    from elspeth.contracts import PluginSchema, RoutingMode

    class SourceOutput(PluginSchema):
        id: int
        name: str

    class SinkA(PluginSchema):
        id: int  # Compatible - only requires id

    class SinkB(PluginSchema):
        id: int
        score: float  # Incompatible - requires field not in source

    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv", output_schema=SourceOutput)
    graph.add_node("gate", node_type="gate", plugin_name="config_gate:fork")  # NO SCHEMA
    graph.add_node("sink_a", node_type="sink", plugin_name="csv_a", input_schema=SinkA)
    graph.add_node("sink_b", node_type="sink", plugin_name="csv_b", input_schema=SinkB)

    graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("gate", "sink_a", label="branch_a", mode=RoutingMode.COPY)  # Fork: COPY mode
    graph.add_edge("gate", "sink_b", label="branch_b", mode=RoutingMode.COPY)  # Fork: COPY mode

    # Should detect incompatibility on gate -> sink_b edge
    with pytest.raises(GraphValidationError) as exc_info:
        graph.validate()

    assert "score" in str(exc_info.value).lower()
    assert "config_gate:fork" in str(exc_info.value)
```

**Step 2: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraph::test_validate_edge_schemas_validates_all_fork_destinations -v
```

Expected: PASS (edge-by-edge validation already handles fork gates correctly)

**Step 3: Commit**

```bash
git add tests/core/test_dag.py
git commit -m "test(dag): verify fork gate edges are validated independently

Fork gates (COPY mode) create multiple outgoing edges. Each edge must be
validated against the effective producer schema. This test confirms that
incompatible fork destinations are caught."
```

---

## Task 9: Update Bug Report Status

**Files:**
- Modify: `docs/bugs/closed/P1-2026-01-21-schema-validator-ignores-dag-routing.md`

**Step 1: Add resolution notes to bug report**

Append to the end of `docs/bugs/closed/P1-2026-01-21-schema-validator-ignores-dag-routing.md`:

```markdown

---

## Resolution

**Status:** RESOLVED (partially - gate routing fixed, aggregations pending)
**Fixed in:** 2026-01-24
**Commit:** [insert commit hash from Task 7]

**Root Cause:**
The edge-based validation was already implemented in `ExecutionGraph._validate_edge_schemas()`, but config-driven gates were added to the graph without `input_schema`/`output_schema` (via `from_config()` method). This caused validation to skip all edges involving gates (check: `if schema is None: continue`).

**Fix Applied:**
Added schema inheritance for gate nodes:
1. Added `get_incoming_edges()` helper to walk backwards through graph
2. Added `_get_effective_producer_schema()` to recursively resolve schema through gate chains
   - Validates multi-input gates have compatible schemas (crashes if not)
   - Crashes if gate has no incoming edges (graph construction bug per CLAUDE.md)
3. Updated `_validate_edge_schemas()` to use effective schema instead of direct attribute access

**Test Coverage:**
- `test_get_incoming_edges_returns_edges_pointing_to_node` - helper method
- `test_get_effective_producer_schema_walks_through_gates` - schema inheritance
- `test_get_effective_producer_schema_crashes_on_gate_without_inputs` - crashes on our bugs
- `test_get_effective_producer_schema_handles_chained_gates` - recursive resolution
- `test_validate_edge_schemas_uses_effective_schema_for_gates` - end-to-end validation
- `test_validate_edge_schemas_validates_all_fork_destinations` - fork gate coverage

**Remaining Work:**
- Aggregation nodes still lack schemas (separate bug needed)
- Coalesce nodes still lack schemas (lower priority)

**Follow-up Bugs:**
- [ ] TODO: Create bug for aggregation schema support
- [ ] TODO: Create bug for coalesce schema support
```

**Step 2: Commit**

```bash
git add docs/bugs/closed/P1-2026-01-21-schema-validator-ignores-dag-routing.md
git commit -m "docs(bugs): document resolution of schema validator DAG routing bug"
```

---

## Task 10: Run Full Test Suite

**Files:**
- None (verification only)

**Step 1: Run engine tests**

```bash
.venv/bin/python -m pytest tests/engine/ -v
```

Expected: All PASS (no engine changes, but verify no cascading failures)

**Step 2: Run core tests**

```bash
.venv/bin/python -m pytest tests/core/ -v
```

Expected: All PASS

**Step 3: Run contract tests**

```bash
.venv/bin/python -m pytest tests/contracts/ -v
```

Expected: All PASS (contracts unchanged)

**Step 4: If any failures, diagnose and fix**

Review failure output. Common issues:
- Import errors: Check module paths
- Assertion errors: Verify expected behavior matches implementation
- Type errors: Check PluginSchema usage

**Step 5: Final commit if fixes needed**

```bash
git add <modified files>
git commit -m "fix: address test failures from schema inheritance changes"
```

---

## Verification Checklist

After completing all tasks:

- [ ] `get_incoming_edges()` returns correct edges for multi-input nodes
- [ ] `_get_effective_producer_schema()` walks through single gate
- [ ] `_get_effective_producer_schema()` crashes on gate with no inputs (CLAUDE.md compliance)
- [ ] `_get_effective_producer_schema()` validates multi-input gates have same schema
- [ ] `_get_effective_producer_schema()` handles chained gates
- [ ] `_get_effective_producer_schema()` returns direct schema for non-gates
- [ ] `_validate_edge_schemas()` catches gate routing to incompatible sink
- [ ] Fork gates (COPY mode) validate all destination edges
- [ ] All existing tests still pass
- [ ] Bug report updated with resolution notes
- [ ] Code follows CLAUDE.md standards (crashes on our bugs, no defensive patterns)

---

## CLAUDE.md Compliance Notes

**Trust Model:**
- Gates are SYSTEM CODE (we own them) - crash on bugs
- Empty incoming edges: CRASH (graph construction bug)
- Incompatible multi-input schemas: CRASH (graph construction bug)
- No defensive patterns: direct attribute access, let it fail if schema missing
- Validation skips `None` schemas (dynamic) but doesn't coerce or default

**Auditability:**
- Schema validation now records "this edge was validated" for gate routes
- Error messages include plugin names for audit trail clarity
- No silent failures - all edges involving gates are validated

**No Legacy Code:**
- No backwards compatibility shims
- Breaking change explicitly noted in commit message
- Old behavior (skipping gate validation) was incorrect - removed completely

---

## Post-Implementation Notes

**For follow-up work:**

1. **Aggregation Schema Support** (separate plan needed):
   - Aggregations have dual schemas (input ≠ output)
   - Requires plugin protocol changes
   - More complex than gate fix

2. **Coalesce Schema Support** (separate plan needed):
   - Output schema depends on merge strategy (union/nested/select)
   - Lower priority - typically followed by permissive output sink

3. **Performance Consideration:**
   - `_get_effective_producer_schema()` is recursive
   - Could be optimized with memoization if gate chains get very deep
   - Current implementation is O(depth) per edge validation call
   - No cycle detection needed - DAG validation runs first

---

## Changes from v1

**Critical fixes based on agent review:**

1. **Removed line number references** - Use semantic anchors ("after get_edges() method")
2. **Fixed multi-input gate handling** - Validates all inputs have compatible schemas, crashes if not
3. **Fixed empty incoming edges** - Crashes per CLAUDE.md (graph construction bug), not defensive None return
4. **Removed inline pytest import** - pytest already imported at module level
5. **Fixed test assertion** - Checks for "config_gate:check" not "csv -> csv"
6. **Added fork gate test** - Validates COPY mode edges to multiple destinations
7. **Clarified existing test limitation** - Existing test uses manually-assigned schemas, doesn't test the bug
8. **Added test for crash on orphan gate** - Verifies CLAUDE.md compliance (crash on our bugs)
