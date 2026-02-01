# Test Defect Report

## Summary
- Gate integration tests validate routing outcomes without verifying Landscape audit records (node_states, token_outcomes, artifacts, hash integrity, lineage) for those flows.

## Severity
- Severity: major
- Priority: P1

## Category
- [Missing Audit Trail Verification]

## Evidence
- `tests/engine/test_engine_gates.py:183` runs `orchestrator.run(...)` and only asserts sink results (`tests/engine/test_engine_gates.py:186`, `tests/engine/test_engine_gates.py:189`) with no audit DB checks.
- `tests/engine/test_engine_gates.py:792` runs the fork test and only checks sink counts/values (`tests/engine/test_engine_gates.py:795`, `tests/engine/test_engine_gates.py:799`), no `node_states`/`token_outcomes`/`artifacts` queries.
- `tests/engine/test_engine_gates.py:1239` queries only `nodes` and asserts registration (`tests/engine/test_engine_gates.py:1243`, `tests/engine/test_engine_gates.py:1252`); no assertions for `node_states`, `token_outcomes`, `artifacts`, hash integrity, or lineage.

```python
result = orchestrator.run(config, graph=_build_test_graph_with_config_gates(config))
assert result.status == "completed"
assert len(match_sink.results) == 1
```

## Impact
- Audit regressions (missing node_states, wrong terminal outcomes, missing artifacts, broken hashes/lineage) can ship while routing tests still pass, violating auditability requirements.
- False confidence in gate coverage because business logic passes without validating the audit trail.

## Root Cause Hypothesis
- Tests focus on routing behavior and lack shared audit-verification helpers, so audit table checks are omitted in most gate flows.

## Recommended Fix
- Extend gate integration tests to query Landscape tables after `orchestrator.run` and assert node_states status/input_hash/output_hash/error, token_outcomes terminal_state/sink_name, and artifacts content_hash/payload_id/artifact_type; for fork tests, assert lineage via `token_parents`/`fork_group_id`.
- Example pattern:
```python
states = conn.execute(
    text("SELECT status, input_hash, output_hash, error_json FROM node_states WHERE node_id = :node_id"),
    {"node_id": gate_node_id},
).fetchall()
assert all(s[0] == "completed" for s in states)
outcomes = conn.execute(
    text("SELECT outcome, sink_name FROM token_outcomes WHERE run_id = :run_id"),
    {"run_id": result.run_id},
).fetchall()
```
- Priority P1 because audit trail completeness is a core ELSPETH requirement.
---
# Test Defect Report

## Summary
- `test_gate_audit_trail_includes_evaluation_metadata` never asserts that the routing event points to the urgent sink; it only checks that a sink node exists.

## Severity
- Severity: minor
- Priority: P2

## Category
- [Weak Assertions]

## Evidence
- `tests/engine/test_engine_gates.py:1409` extracts `to_node_id`, but the test only asserts a non-None node (`tests/engine/test_engine_gates.py:1417`) and leaves a comment instead of a check (`tests/engine/test_engine_gates.py:1418`).

```python
to_node_id = routing_event[5]
sink_node = conn.execute(...).fetchone()
assert sink_node is not None
# The sink plugin name should indicate it's the urgent sink
```

## Impact
- Misrouted edges or incorrect sink mapping in audit records can pass unnoticed, weakening audit trail verification for gate routing.

## Root Cause Hypothesis
- Assertion was left incomplete (commented intent without enforcement).

## Recommended Fix
- Assert the expected sink node explicitly, e.g., `assert to_node_id == "sink_urgent"` based on `_build_test_graph_with_config_gates` naming, and also validate the "continue" routing event.
- Example:
```python
assert to_node_id == "sink_urgent"
assert any(e[4] == "continue" for e in routing_events)
```
- Priority P2 because it weakens a critical audit test but does not block core execution.
---
# Test Defect Report

## Summary
- Inline `ListSource` and `CollectSink` classes are duplicated across many tests, indicating fixture duplication.

## Severity
- Severity: trivial
- Priority: P3

## Category
- [Fixture Duplication]

## Evidence
- `tests/engine/test_engine_gates.py:206` and `tests/engine/test_engine_gates.py:220` define `ListSource`/`CollectSink` in `test_composite_or_condition`.
- `tests/engine/test_engine_gates.py:724` and `tests/engine/test_engine_gates.py:738` repeat the same `ListSource`/`CollectSink` definitions in fork tests.

```python
class ListSource(_TestSourceBase):
    name = "list_source"
    output_schema = InputSchema
    ...
class CollectSink(_TestSinkBase):
    name = "collect"
    config: ClassVar[dict[str, Any]] = {}
    ...
```

## Impact
- Higher maintenance cost and risk of inconsistent updates when source/sink behavior changes.
- Makes it harder to enforce consistent audit assertions across tests.

## Root Cause Hypothesis
- Convenience inline definitions without shared fixtures or helper factories.

## Recommended Fix
- Factor shared test sources/sinks into pytest fixtures or helper factories at the top of this file (or `tests/conftest.py`) and reuse across tests.
- Example:
```python
def make_list_source(schema, data):
    class ListSource(_TestSourceBase):
        name = "list_source"
        output_schema = schema
        def __init__(self, data): self._data = data
        def load(self, ctx): yield from (SourceRow.valid(r) for r in self._data)
    return ListSource(data)
```
- Priority P3 because this is a maintainability issue rather than a functional test gap.
