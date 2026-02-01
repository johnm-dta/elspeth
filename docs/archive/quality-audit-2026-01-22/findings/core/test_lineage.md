# Test Quality Review: test_lineage.py

## Summary

The lineage tests verify basic query routing and error handling but **completely fail to validate the core auditability contract**: complete lineage tracking from source → transforms → sinks with hash verification. Tests are shallow smoke tests that verify method calls succeed without checking the correctness of lineage data assembly. Critical gaps include no transform lineage, no hash verification, no parent token traversal, and no end-to-end pipeline scenarios.

## Poorly Constructed Tests

### Test: test_lineage_result_fields (line 20)
**Issue**: Trivial constructor smoke test with hardcoded values, provides zero validation of contract invariants.
**Evidence**: Manually constructs `LineageResult` with dummy data (`token_id="t1"`, `row_id="r1"`), asserts the exact values it just set. Does not verify:
- Source row hash integrity
- Node states ordered by `step_index`
- Routing events linked to correct states
- Parent token resolution
- Payload availability semantics

**Fix**: Delete this test. Constructor validation is useless. If you need structural validation, write property tests for `LineageResult` invariants (e.g., "all routing events reference valid state_ids").
**Priority**: P3

### Test: test_explain_returns_lineage_result (line 61)
**Issue**: Only verifies that `explain()` returns a non-None object with correct IDs. Does not validate lineage correctness.
**Evidence**:
```python
assert isinstance(result, LineageResult)
assert result.token.token_id == token.token_id
assert result.source_row.row_id == row.row_id
```
Missing validations:
- `source_row.source_data_hash` matches recorded hash
- `source_row.source_data` matches original `{"id": 1}`
- `source_row.payload_available` is True
- `node_states` contains the source node
- `calls`, `routing_events`, `parent_tokens` are empty (since no transforms)

**Fix**: Expand to validate the **complete lineage contract** for a minimal pipeline (source → no transforms → no sink). Verify all fields are populated correctly.
**Priority**: P1

### Test: test_explain_by_row_id (line 96)
**Issue**: Creates a terminal outcome solely to satisfy `explain()` logic, then asserts `result is not None`. Does not verify lineage data.
**Evidence**: Records `COMPLETED` outcome at `sink_name="output"` but never validates:
- The outcome is included in `result.outcome`
- The outcome's `sink_name` is correct
- The token reached terminal state

**Fix**: Verify `result.outcome.outcome == RowOutcome.COMPLETED`, `result.outcome.sink_name == "output"`, and validate source row hash.
**Priority**: P2

### Test: test_explain_nonexistent_returns_none (line 139)
**Issue**: Correct behavior test but should also verify graceful handling of partial data scenarios (e.g., token exists but row deleted, row exists but token quarantined).
**Evidence**: Only tests fully nonexistent `run_id` and `token_id`.
**Fix**: Add edge cases: token exists but row missing (corruption scenario), row exists but no terminal tokens yet (in-progress scenario).
**Priority**: P2

### Test: test_explain_fork_with_sink_disambiguation (line 151)
**Issue**: Tests disambiguation logic but does not verify **lineage correctness** for forked tokens. Missing parent token validation.
**Evidence**: Creates two tokens from one row, routes to different sinks, asserts token_id matches. Does not verify:
- `result_a.parent_tokens` and `result_b.parent_tokens` are empty (no fork parent recorded)
- Both tokens share the same `source_row.source_data_hash`
- Routing events for each token are distinct

**Fix**: Add assertions for parent tokens (should be empty for these scenarios), verify shared source row hash, validate routing events match the expected paths.
**Priority**: P1

### Test: test_explain_buffered_tokens_returns_none (line 295)
**Issue**: Correct behavior test but does not validate the **reason** for returning None (all tokens non-terminal vs. row not found).
**Evidence**: Records `BUFFERED` outcome and asserts `result is None`. Does not distinguish between "row not found" and "row exists but all tokens non-terminal."
**Fix**: Query outcomes directly and verify `len(terminal_outcomes) == 0` to confirm buffered tokens exist.
**Priority**: P3

### Test: test_explain_multiple_tokens_same_sink_raises (line 334)
**Issue**: Verifies error message but does not test **why** multiple tokens at the same sink is invalid (expand scenario detection).
**Evidence**: Creates 3 tokens routed to `same_sink`, asserts ValueError with message match. Does not:
- Explain when this scenario is valid (expand transform) vs. invalid (pipeline misconfiguration)
- Verify the error message lists all ambiguous token IDs

**Fix**: Document that this tests the "expand transform" scenario where a single row produces multiple outputs at the same sink. Verify error message contains all 3 token IDs.
**Priority**: P3

## Misclassified Tests

### Test: test_explain_returns_lineage_result (line 61)
**Issue**: Claims to be a unit test but constructs an in-memory database and uses `LandscapeRecorder` (integration test scope).
**Evidence**: Uses `LandscapeDB.in_memory()` and `LandscapeRecorder(db)` - this is integration-level testing.
**Fix**: All tests in this file should be classified as **integration tests** and moved to `tests/integration/test_lineage_query.py`. These tests require database state and recorder orchestration.
**Priority**: P2

### All tests in TestExplainFunction (line 52)
**Issue**: Every test is an integration test, not a unit test. They all construct database + recorder + run state.
**Evidence**: Every test calls `LandscapeDB.in_memory()`, `recorder.begin_run()`, `recorder.create_row()`, etc.
**Fix**: Rename file to `test_lineage_integration.py` and move to `tests/integration/`. Create separate `tests/core/landscape/test_lineage_unit.py` for true unit tests (e.g., `LineageResult` validation logic, query parameter validation without database).
**Priority**: P2

## Infrastructure Gaps

### Gap: No fixtures for common setup (P1)
**Issue**: Every test duplicates the same setup: create DB, recorder, run, source node, row, token. 20+ lines of identical setup per test.
**Evidence**: Lines 68-88, 103-121, 160-178, 220-238, etc. all repeat:
```python
db = LandscapeDB.in_memory()
recorder = LandscapeRecorder(db)
run = recorder.begin_run(...)
node = recorder.register_node(...)
```

**Fix**: Create pytest fixtures:
```python
@pytest.fixture
def recorder():
    db = LandscapeDB.in_memory()
    return LandscapeRecorder(db)

@pytest.fixture
def minimal_run(recorder):
    run = recorder.begin_run(config={}, canonical_version="sha256-rfc8785-v1")
    node = recorder.register_node(...)
    return run, node

@pytest.fixture
def single_row(recorder, minimal_run):
    run, node = minimal_run
    row = recorder.create_row(...)
    token = recorder.create_token(row_id=row.row_id)
    return row, token
```

### Gap: No transform lineage tests (P0 - CRITICAL)
**Issue**: **ZERO tests verify lineage through transforms**. The core auditability contract is "source → transform → sink" lineage, yet no test validates:
- Transform node states in `result.node_states`
- Transform input/output hashes in node states
- Transform sequence ordering (step_index)
- Transform errors in `result.transform_errors`
- External calls in `result.calls`

**Evidence**: All tests use source-only pipelines. The Attributability Test from CLAUDE.md requires:
```python
lineage = landscape.explain(run_id, token_id=token_id, field=field)
assert lineage.source_row is not None
assert len(lineage.node_states) > 0  # Should include transforms!
```

**Fix**: Add test suite `test_explain_with_transforms`:
```python
def test_explain_linear_pipeline_with_two_transforms():
    # source → transform_a → transform_b → sink
    # Verify:
    # - len(result.node_states) == 3 (source + 2 transforms)
    # - node_states[0].node_type == NodeType.SOURCE
    # - node_states[1].node_type == NodeType.TRANSFORM
    # - node_states[2].node_type == NodeType.TRANSFORM
    # - step_index ordering: [0, 1, 2]
    # - input_hash of transform_b == output_hash of transform_a
    # - final output_hash matches artifact hash at sink
```

### Gap: No hash verification tests (P0 - CRITICAL)
**Issue**: Hash integrity is the foundation of auditability ("Hashes survive payload deletion"), yet no test verifies:
- `source_row.source_data_hash` matches the hash of `source_row.source_data`
- Transform `input_hash` matches previous transform's `output_hash`
- Artifact `content_hash` matches final node state `output_hash`
- Hash chain is unbroken through the pipeline

**Evidence**: Test line 94 asserts `result.source_row.row_id == row.row_id` but never checks `source_row.source_data_hash` correctness.

**Fix**: Add property test:
```python
@given(st.dictionaries(st.text(), st.integers()))
def test_source_row_hash_matches_data(data):
    # Create row with arbitrary data
    # Verify source_row.source_data_hash == canonical_hash(data)
```
Add integration test for transform hash chain:
```python
def test_transform_hash_chain_integrity():
    # source → transform_a → transform_b → sink
    # Verify:
    # - transform_a.input_hash == source.output_hash
    # - transform_b.input_hash == transform_a.output_hash
    # - artifact.content_hash == transform_b.output_hash
```

### Gap: No parent token lineage tests (P1)
**Issue**: DAG forks/joins use parent tokens (`parent_token_id`) for lineage, but no test validates parent token traversal.
**Evidence**: `test_explain_fork_with_sink_disambiguation` creates forked tokens but never asserts `result.parent_tokens` content. The lineage implementation (line 168-173) fetches parent tokens, but no test validates this works.

**Fix**: Add test:
```python
def test_explain_includes_parent_tokens_for_coalesce():
    # Create fork: row → [token_a, token_b] → coalesce → token_merged
    # Set token_merged.parent_token_id = [token_a.token_id, token_b.token_id]
    # Verify:
    # - len(result.parent_tokens) == 2
    # - result.parent_tokens contains token_a and token_b
    # - Each parent token has correct row_id
```

### Gap: No validation error tests (P1)
**Issue**: `LineageResult` includes `validation_errors` field, but no test verifies validation errors are attached to lineage.
**Evidence**: Implementation line 176 fetches validation errors by hash, but no test creates validation errors and checks they appear in lineage.

**Fix**: Add test:
```python
def test_explain_includes_validation_errors():
    # Create row with validation errors (e.g., schema mismatch)
    # Record validation error via recorder
    # Verify:
    # - len(result.validation_errors) > 0
    # - validation_error.field_name == expected
    # - validation_error.error_message matches expected
```

### Gap: No external call lineage tests (P1)
**Issue**: `LineageResult.calls` should contain all external calls (LLM, API, DB) made during transform processing. Zero tests validate this.
**Evidence**: Implementation line 162-165 fetches calls per state, but no test creates calls and verifies they appear in lineage.

**Fix**: Add test:
```python
def test_explain_includes_external_calls():
    # source → llm_transform (records external call) → sink
    # Record a call via recorder.record_call(state_id, call_type, request, response)
    # Verify:
    # - len(result.calls) == 1
    # - result.calls[0].call_type == CallType.LLM
    # - result.calls[0].request_hash is not None
    # - result.calls[0].response_hash is not None
```

### Gap: No routing event tests (P2)
**Issue**: Gates produce routing events, but no test verifies routing events are correctly attached to lineage.
**Evidence**: Implementation line 156-159 fetches routing events, but no test creates routing events (e.g., gate routing to different sinks) and validates they appear.

**Fix**: Add test:
```python
def test_explain_includes_routing_events():
    # source → gate (routes to sink_a vs sink_b based on field) → sinks
    # Record routing event via recorder
    # Verify:
    # - len(result.routing_events) > 0
    # - routing_event.routing_kind == RoutingKind.SINK_ROUTE
    # - routing_event.destination matches expected sink
```

### Gap: No end-to-end pipeline tests (P0 - CRITICAL)
**Issue**: No test validates **complete lineage** for a realistic pipeline (source → transform → gate → sink).
**Evidence**: All tests use minimal source-only pipelines. The auditability standard requires:
> "For any output, the system must prove complete lineage: source data, configuration, and code version."

**Fix**: Add integration test:
```python
def test_explain_complete_pipeline_lineage():
    # Pipeline: csv_source → field_mapper → classification_gate → [approved_sink, rejected_sink]
    # Process one row through to completion
    # Verify lineage includes:
    # 1. Source row with correct hash
    # 2. Node states: [source, field_mapper, classification_gate, sink]
    # 3. Transform at each step with input/output hashes
    # 4. Routing event showing gate decision
    # 5. Terminal outcome at correct sink
    # 6. Hash chain integrity from source → transforms → artifact
```

### Gap: No property-based tests (P2)
**Issue**: Lineage queries should be deterministic and idempotent, but no property tests validate this.
**Evidence**: No use of Hypothesis for generative testing. CLAUDE.md recommends Hypothesis for "avoiding manual edge-case hunting."

**Fix**: Add property tests:
```python
@given(st.integers(min_value=1, max_value=100))
def test_explain_idempotent(num_queries):
    # Call explain() N times for same token_id
    # Verify all results are identical (deep equality)

@given(st.data())
def test_explain_deterministic_for_any_pipeline(data):
    # Generate random pipeline structure
    # Verify explain() always returns same lineage for same token
```

## Test Architecture Issues

### Issue: No test data builders (P2)
**Problem**: Tests manually construct complex state (run, node, row, token, outcome) with hardcoded values, making them brittle and hard to read.
**Fix**: Create builder pattern for test data:
```python
class LineageTestBuilder:
    def with_source(self, plugin_name="csv", data=None):
        ...
    def with_transform(self, plugin_name, input_data, output_data):
        ...
    def with_terminal_outcome(self, outcome, sink_name):
        ...
    def build(self) -> LineageResult:
        ...

# Usage:
result = (
    LineageTestBuilder(recorder)
    .with_source(data={"id": 1})
    .with_transform("mapper", input_data={"id": 1}, output_data={"id": "1"})
    .with_terminal_outcome(RowOutcome.COMPLETED, "output")
    .build()
)
```

### Issue: Tests don't verify invariants (P1)
**Problem**: Tests check specific values but don't validate structural invariants that should always hold:
- `node_states` sorted by `step_index`
- All routing events reference valid `state_id` values
- Parent tokens exist if `parent_token_id` is set
- `payload_available=False` means `source_data` is None

**Fix**: Create invariant validator:
```python
def validate_lineage_invariants(result: LineageResult) -> list[str]:
    """Return list of invariant violations."""
    violations = []

    # Check step_index ordering
    step_indices = [s.step_index for s in result.node_states]
    if step_indices != sorted(step_indices):
        violations.append(f"step_index not ordered: {step_indices}")

    # Check routing events reference valid states
    state_ids = {s.state_id for s in result.node_states}
    for event in result.routing_events:
        if event.state_id not in state_ids:
            violations.append(f"routing event {event} references unknown state_id")

    # Check payload availability
    if not result.source_row.payload_available and result.source_row.source_data is not None:
        violations.append("payload_available=False but source_data is not None")

    return violations

# Use in tests:
result = explain(recorder, run_id=run.run_id, token_id=token.token_id)
violations = validate_lineage_invariants(result)
assert not violations, f"Invariant violations: {violations}"
```

## Positive Observations

- **Error handling is tested**: Tests cover `ValueError` for ambiguous queries (`test_explain_fork_with_sink_disambiguation`, `test_explain_multiple_tokens_same_sink_raises`).
- **Edge cases for disambiguation**: Tests validate sink disambiguation logic for forked rows.
- **None return semantics**: Tests correctly verify when `explain()` should return None (nonexistent data, buffered tokens, wrong sink).

## Critical Missing Coverage Summary

| Missing Coverage | Priority | Impact |
|------------------|----------|--------|
| Transform lineage validation | P0 | **Cannot prove auditability contract** - core feature untested |
| Hash chain integrity | P0 | **Cannot prove data integrity** - hashes never verified |
| End-to-end pipeline lineage | P0 | **No real-world scenario tested** - only trivial pipelines |
| Parent token traversal | P1 | Fork/join lineage broken if this fails |
| External call recording | P1 | LLM/API calls invisible in audit trail |
| Validation errors in lineage | P1 | Source validation failures not traceable |
| Routing events in lineage | P2 | Gate decisions not auditable |
| Property-based testing | P2 | Idempotence/determinism not validated |

## Recommendations

### Immediate (P0)
1. **Create `test_lineage_transform_chain.py`**: Test source → transform_a → transform_b → sink with complete hash verification.
2. **Add hash verification to all existing tests**: Assert `source_data_hash` matches canonical hash of `source_data`.
3. **Create end-to-end pipeline test**: CSV source → field mapper → gate → multiple sinks, verify complete lineage.

### Short-term (P1)
4. **Create pytest fixtures**: Eliminate duplicated setup (recorder, run, node, row).
5. **Add parent token tests**: Verify fork/join lineage with parent token traversal.
6. **Add external call tests**: Verify LLM/API calls appear in lineage.
7. **Add validation error tests**: Verify source validation failures traceable in lineage.

### Medium-term (P2)
8. **Reclassify tests**: Move to `tests/integration/` and create true unit tests for `LineageResult` validation.
9. **Add property-based tests**: Use Hypothesis to verify idempotence and determinism.
10. **Create test data builders**: Builder pattern for readable, maintainable test setup.
11. **Add invariant validators**: Structural invariants checked in all tests.

### Test Template for Missing Coverage

```python
def test_explain_complete_audit_trail():
    """Verify complete lineage from source through transforms to sink.

    This is the core auditability contract: every output must be traceable
    to source data, configuration, and code version with hash verification.
    """
    # Setup: source → transform_a → transform_b → sink
    recorder = ...
    run = ...
    source_node = recorder.register_node(plugin_name="csv", ...)
    transform_a = recorder.register_node(plugin_name="field_mapper", ...)
    transform_b = recorder.register_node(plugin_name="classifier", ...)
    sink_node = recorder.register_node(plugin_name="json_sink", ...)

    # Process row through pipeline
    row = recorder.create_row(data={"id": 1, "value": "test"})
    token = recorder.create_token(row_id=row.row_id)

    # Record state at each step with hash chain
    state_source = recorder.record_node_state(
        token_id=token.token_id,
        node_id=source_node.node_id,
        input_hash=None,
        output_hash="hash_source_out",
        output_data={"id": 1, "value": "test"},
    )
    state_a = recorder.record_node_state(
        token_id=token.token_id,
        node_id=transform_a.node_id,
        input_hash="hash_source_out",  # Chain continues
        output_hash="hash_a_out",
        output_data={"id": "1", "value": "test"},  # Type transform
    )
    state_b = recorder.record_node_state(
        token_id=token.token_id,
        node_id=transform_b.node_id,
        input_hash="hash_a_out",  # Chain continues
        output_hash="hash_b_out",
        output_data={"id": "1", "value": "test", "class": "approved"},
    )

    recorder.record_token_outcome(
        token_id=token.token_id,
        outcome=RowOutcome.COMPLETED,
        sink_name="approved_output",
    )

    # Query lineage
    result = explain(recorder, run_id=run.run_id, token_id=token.token_id)

    # VERIFY COMPLETE AUDIT TRAIL
    assert result is not None

    # 1. Source row with hash
    assert result.source_row.row_id == row.row_id
    assert result.source_row.source_data == {"id": 1, "value": "test"}
    assert result.source_row.source_data_hash == canonical_hash({"id": 1, "value": "test"})

    # 2. Node states in order
    assert len(result.node_states) == 3
    assert result.node_states[0].node_id == source_node.node_id
    assert result.node_states[1].node_id == transform_a.node_id
    assert result.node_states[2].node_id == transform_b.node_id
    assert [s.step_index for s in result.node_states] == [0, 1, 2]

    # 3. Hash chain integrity
    assert result.node_states[0].output_hash == "hash_source_out"
    assert result.node_states[1].input_hash == "hash_source_out"  # Chain link
    assert result.node_states[1].output_hash == "hash_a_out"
    assert result.node_states[2].input_hash == "hash_a_out"  # Chain link
    assert result.node_states[2].output_hash == "hash_b_out"

    # 4. Terminal outcome
    assert result.outcome is not None
    assert result.outcome.outcome == RowOutcome.COMPLETED
    assert result.outcome.sink_name == "approved_output"

    # 5. Attributability: Can explain every decision
    # (In real usage, this would query field-level lineage)
    assert result.source_row is not None  # Can trace back to source
    assert len(result.node_states) > 0  # Can trace through transforms
    assert result.outcome is not None  # Can explain final destination
```
