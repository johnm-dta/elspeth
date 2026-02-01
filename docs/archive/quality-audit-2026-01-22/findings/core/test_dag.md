# Test Quality Review: test_dag.py

## Summary

The test file is comprehensive (2316 lines, 93 tests) but suffers from significant structural issues: excessive reliance on integration fixtures making tests slow and fragile, poor isolation with shared mutable state, 15 skipped tests indicating incomplete implementation, missing property-based testing for DAG invariants, and several tests that don't verify NetworkX delegation despite CLAUDE.md's explicit requirement to "avoid reinventing graph algorithms."

## Poorly Constructed Tests

### Test: test_from_config_minimal (line 501)
**Issue**: Integration test disguised as unit test - instantiates entire plugin system
**Evidence**:
```python
plugins = instantiate_plugins_from_config(config)
graph = ExecutionGraph.from_plugin_instances(
    source=plugins["source"],
    transforms=plugins["transforms"],
    sinks=plugins["sinks"],
    aggregations=plugins["aggregations"],
    gates=list(config.gates),
    output_sink=config.output_sink,
)
```
**Fix**: Create a unit test fixture that returns mock plugin instances. Test `from_plugin_instances()` separately from plugin instantiation.
**Priority**: P2

### Test: test_from_config_is_valid (line 540)
**Issue**: Duplicate of test_from_config_minimal - both tests instantiate plugins, build graph, and validate
**Evidence**: Nearly identical setup to test_from_config_minimal, only difference is calling `validate()` at end
**Fix**: Delete this test and add `graph.validate()` assertion to test_from_config_minimal
**Priority**: P3

### Test: test_from_config_with_transforms (line 577)
**Issue**: Fragile substring matching for topological order validation
**Evidence**:
```python
assert "source" in order[0]  # Substring match!
assert "sink" in order[-1]    # Substring match!
passthrough_idx = next(i for i, n in enumerate(order) if "passthrough" in n)
field_mapper_idx = next(i for i, n in enumerate(order) if "field_mapper" in n)
```
**Fix**: Use the explicit node ID accessors (`get_source()`, `get_sink_id_map()`, `get_transform_id_map()`) to get exact IDs, then verify topology
**Priority**: P2

### Test: test_from_config_with_gate_routes (line 631)
**Issue**: Assertion counts nodes without verifying critical topology structure
**Evidence**:
```python
assert graph.node_count == 4  # source, config_gate, results, flagged
assert graph.edge_count == 3
```
**Fix**: Verify gate actually routes to both sinks by checking edges explicitly:
```python
edges = graph.get_edges()
gate_edges = [e for e in edges if e.from_node == gate_id]
assert {e.to_node for e in gate_edges} == {results_id, flagged_id}
```
**Priority**: P2

### Test: test_validate_raises_on_cycle (line 79)
**Issue**: Tests exception message match but doesn't verify the cycle is correctly identified
**Evidence**:
```python
with pytest.raises(GraphValidationError, match="cycle"):
    graph.validate()
```
**Fix**: Check that the error message contains the actual cycle nodes (a, b):
```python
with pytest.raises(GraphValidationError) as exc_info:
    graph.validate()
assert "a" in str(exc_info.value) and "b" in str(exc_info.value)
```
**Priority**: P3

### Test: test_topological_order (line 91)
**Issue**: Tests only partial ordering constraints, not full topological sort correctness
**Evidence**: Only checks source first, sink last, and t1 before t2. Doesn't verify t1/t2 both come before sink.
**Fix**: Verify all ordering constraints:
```python
assert order.index("source") < order.index("t1") < order.index("t2") < order.index("sink")
```
**Priority**: P3

### Test: test_get_incoming_edges_returns_edges_pointing_to_node (line 285)
**Issue**: Test name is 83 characters - violates readability
**Evidence**: Full name is `test_get_incoming_edges_returns_edges_pointing_to_node`
**Fix**: Rename to `test_get_incoming_edges`
**Priority**: P3

### Test: test_partial_branch_coverage_branches_not_in_coalesce_route_to_sink (line 1323)
**Issue**: Test name is 81 characters and test does too much (verifies both coalesce AND non-coalesce routing)
**Evidence**: Single test verifies path_c routes to sink AND path_a/path_b route to coalesce
**Fix**: Split into two tests:
- `test_uncovered_fork_branches_route_to_sink` (verify path_c only)
- `test_coalesced_branches_route_to_coalesce_node` (verify path_a/path_b only)
**Priority**: P2

### Test: test_from_plugin_instances_extracts_schemas (line 1945)
**Issue**: Creates temporary file but accesses private graph internals
**Evidence**:
```python
source_nodes = [n for n, d in graph._graph.nodes(data=True) if d["info"].node_type == "source"]
source_info = graph.get_node_info(source_nodes[0])
```
**Fix**: Use public API only:
```python
source_id = graph.get_source()
source_info = graph.get_node_info(source_id)
assert source_info.output_schema is not None
```
**Priority**: P2

### Test: test_node_ids_are_deterministic_for_same_config (line 2197)
**Issue**: Builds entire plugin system twice but only checks node ID equality - expensive for what it verifies
**Evidence**: Instantiates plugins, builds full graphs, then just compares sorted node lists
**Fix**: Extract to integration test directory or mock plugin instantiation
**Priority**: P2

### Test: All tests in TestExecutionGraphFromConfig class (lines 498-849)
**Issue**: Entire class depends on `plugin_manager` fixture - makes tests slow and couples DAG logic to plugin system
**Evidence**: Every test method signature includes `plugin_manager` parameter
**Fix**: Create mock plugin instances in a `@pytest.fixture` that doesn't invoke the real plugin system:
```python
@pytest.fixture
def mock_source():
    return Mock(spec=SourceProtocol, plugin_name="csv", ...)
```
**Priority**: P1

## Misclassified Tests

### Test: test_from_config_validates_route_targets (line 682)
**Issue**: This is an error path test (config validation) mixed with DAG construction tests
**Evidence**: Expects `GraphValidationError` for nonexistent sink reference - this is config validation, not graph validation
**Fix**: Move to `tests/core/test_config.py` or create `tests/core/test_config_validation.py`
**Priority**: P2

### Test: test_from_plugin_instances_extracts_schemas (line 1945)
**Issue**: Tests schema extraction from plugin instances - this is plugin protocol testing, not DAG testing
**Evidence**: Uses tempfile to load config, instantiate plugins, then just verifies schema was copied to NodeInfo
**Fix**: Move to `tests/plugins/test_plugin_protocols.py` or similar
**Priority**: P2

### Test: TestDeterministicNodeIDs class (lines 2194-2316)
**Issue**: Tests node ID hashing/generation which belongs in checkpoint compatibility tests
**Evidence**: Tests explicitly document "for checkpoint/resume compatibility" but are in DAG test file
**Fix**: Move to `tests/integration/test_checkpoint_node_id_stability.py`
**Priority**: P2

## Infrastructure Gaps

### Gap: Missing pytest fixture for mock ExecutionGraph
**Issue**: Every test manually constructs graphs with repetitive `add_node` / `add_edge` calls
**Evidence**: 50+ tests repeat this pattern:
```python
graph = ExecutionGraph()
graph.add_node("source", node_type="source", plugin_name="csv")
graph.add_node("sink", node_type="sink", plugin_name="csv")
graph.add_edge("source", "sink", label="continue")
```
**Fix**: Create fixtures:
```python
@pytest.fixture
def simple_graph():
    """Source -> Sink"""
    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv")
    graph.add_node("sink", node_type="sink", plugin_name="csv")
    graph.add_edge("source", "sink", label="continue")
    return graph

@pytest.fixture
def linear_graph():
    """Source -> T1 -> T2 -> Sink"""
    ...
```
**Priority**: P1

### Gap: Missing property-based tests for DAG invariants
**Issue**: No Hypothesis tests despite CLAUDE.md requiring property testing and DAG having well-defined invariants
**Evidence**: CLAUDE.md lists "Property Testing: Hypothesis" in acceleration stack; DAG has clear invariants:
- Acyclicity (no path from node to itself)
- Reachability (all nodes reachable from source)
- Single source, at least one sink
- Topological sort exists iff graph is acyclic
**Fix**: Add property tests:
```python
from hypothesis import given
from hypothesis.strategies import builds

@given(dag=builds(valid_dag_strategy))
def test_topological_sort_respects_edges(dag):
    """If edge A->B exists, A appears before B in any topological sort."""
    order = dag.topological_order()
    for edge in dag.get_edges():
        assert order.index(edge.from_node) < order.index(edge.to_node)
```
**Priority**: P1

### Gap: Missing verification of NetworkX delegation
**Issue**: Tests verify behavior but don't verify that NetworkX is actually being used
**Evidence**: CLAUDE.md explicitly states "DAG Validation: NetworkX - Custom graph algorithms (acyclicity, topo sort)" but no test verifies `nx.is_directed_acyclic_graph()` or `nx.topological_sort()` are called
**Fix**: Add tests that mock NetworkX functions to ensure delegation:
```python
@patch('elspeth.core.dag.nx.is_directed_acyclic_graph')
def test_is_acyclic_delegates_to_networkx(mock_nx_is_dag):
    mock_nx_is_dag.return_value = True
    graph = ExecutionGraph()
    result = graph.is_acyclic()
    mock_nx_is_dag.assert_called_once_with(graph._graph)
```
**Priority**: P2

### Gap: Shared mutable state in TestCoalesceNodes
**Issue**: Multiple tests in this class modify the same graph structure without isolation
**Evidence**: Tests at lines 1190, 1254, 1398, 1470, 1530, 1596 all build graphs with fork/coalesce but don't use fixtures - if one test mutates the graph, others could fail
**Fix**: Extract common graph building to a fixture with `autouse=False` and explicit invocation per test
**Priority**: P2

### Gap: Missing edge case tests for MultiDiGraph
**Issue**: MultiDiGraph allows parallel edges but only happy-path cases tested
**Evidence**: Tests verify multiple edges exist (line 1024) but don't test:
- What happens with 100 parallel edges to same destination?
- Edge iteration order stability
- Edge deletion with duplicate labels
**Fix**: Add edge case tests:
```python
def test_many_parallel_edges_performance():
    """Verify MultiDiGraph doesn't degrade with many parallel edges."""
    graph = ExecutionGraph()
    graph.add_node("gate", ...)
    graph.add_node("sink", ...)
    for i in range(1000):
        graph.add_edge("gate", "sink", label=f"route_{i}")
    assert graph.edge_count == 1000
```
**Priority**: P3

### Gap: No tests for graph traversal or reachability
**Issue**: DAG execution requires path finding but no tests verify reachability queries
**Evidence**: CLAUDE.md mentions "Path finding for lineage queries" but no tests verify:
- All nodes reachable from source
- Unreachable nodes detection
- Path enumeration between nodes
**Fix**: Add reachability tests:
```python
def test_unreachable_node_detected():
    """Orphaned nodes should be detected during validation."""
    graph = ExecutionGraph()
    graph.add_node("source", node_type="source", plugin_name="csv")
    graph.add_node("orphan", node_type="transform", plugin_name="x")
    graph.add_node("sink", node_type="sink", plugin_name="csv")
    graph.add_edge("source", "sink", label="continue")
    # orphan is not connected - validation should fail
    with pytest.raises(GraphValidationError, match="unreachable"):
        graph.validate()
```
**Priority**: P1

### Gap: Skipped tests without clear restoration plan
**Issue**: 15 tests marked `@pytest.mark.skip(reason="Method deleted in Task 2, will be restored in Task 2.5")`
**Evidence**: Lines 318, 342, 357, 406, 427, 463, 2013, 2051, 1666, 1747, 1865, 1907, 2163
**Fix**: Either:
1. Complete Task 2.5 and restore the methods, OR
2. Delete the tests if the functionality is permanently removed
**Priority**: P1

### Gap: Missing negative test for duplicate node IDs
**Issue**: Tests verify nodes can be added but don't test duplicate node ID rejection
**Evidence**: `add_node()` doesn't check for duplicates - what happens if called twice with same node_id?
**Fix**: Add test:
```python
def test_add_node_rejects_duplicate_id():
    graph = ExecutionGraph()
    graph.add_node("n1", node_type="source", plugin_name="csv")
    with pytest.raises(GraphValidationError, match="duplicate"):
        graph.add_node("n1", node_type="sink", plugin_name="csv")
```
**Priority**: P2

### Gap: No test coverage for edge mode validation
**Issue**: Tests use RoutingMode.MOVE and RoutingMode.COPY but don't verify invalid modes are rejected
**Evidence**: Multiple tests pass `mode=RoutingMode.MOVE` but none verify what happens with invalid mode
**Fix**: Add test:
```python
def test_add_edge_rejects_invalid_mode():
    graph = ExecutionGraph()
    graph.add_node("a", ...)
    graph.add_node("b", ...)
    with pytest.raises(ValueError):
        graph.add_edge("a", "b", label="test", mode="INVALID")
```
**Priority**: P3

## Positive Observations

- Excellent test coverage breadth: Tests cover basic graph construction, validation, config integration, multi-edge scenarios, coalesce nodes, schema detection, and deterministic node IDs.
- Good use of descriptive test class names that group related functionality (TestDAGBuilder, TestDAGValidation, TestSourceSinkValidation, etc.).
- Comprehensive fork/join testing: TestCoalesceNodes class thoroughly exercises the complex fork/join DAG scenarios.
- Tests explicitly verify MultiDiGraph behavior (parallel edges, label preservation) which is critical for the routing model.
- Strong coverage of negative cases: Tests verify cycle detection, missing source/sink, duplicate edge labels, invalid route targets.
- Tests include regression tests with clear documentation (e.g., test_hyphenated_sink_names_work_in_dag at line 952).
