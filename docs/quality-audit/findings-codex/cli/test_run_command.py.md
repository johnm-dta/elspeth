# Test Defect Report

## Summary

- CLI run tests execute pipelines but do not validate the Landscape audit trail tables or hash/lineage integrity despite configuring a Landscape DB

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/cli/test_run_command.py:54` runs the pipeline and `tests/cli/test_run_command.py:60` only asserts output file existence, while `tests/cli/test_run_command.py:48` configures a Landscape DB; no audit DB checks are performed.
```python
result = runner.invoke(app, ["run", "--settings", str(pipeline_settings), "--execute"])
assert result.exit_code == 0

output_file = tmp_path / "output.json"
assert output_file.exists()
```
- `tests/cli/test_run_command.py:586` only queries `nodes_table`; there are no assertions for `node_states`, `token_outcomes`, `artifacts`, hash integrity, or lineage.
```python
nodes = conn.execute(select(nodes_table)).fetchall()
recorded_node_ids = {node.node_id for node in nodes}
```
- `src/elspeth/cli.py:565` shows the CLI execution path explicitly creates audit trail records via `Orchestrator` + `LandscapeDB`, so tests should verify those writes.

## Impact

- Audit trail regressions (missing node_states, token_outcomes, artifacts, or hash/lineage corruption) can ship undetected.
- Violates the Auditability Standard’s requirement that “if it’s not recorded, it didn’t happen.”
- Creates false confidence: pipelines can appear to run correctly while audit tables are incomplete or wrong.

## Root Cause Hypothesis

- CLI tests focus on output files and console output, with audit verification treated as out-of-scope or assumed covered elsewhere.
- No shared helper/fixture for querying Landscape tables, so audit checks are omitted.

## Recommended Fix

- Extend the `--execute` tests in `tests/cli/test_run_command.py` to open the Landscape DB and assert:
  - `node_states_table` has expected statuses with non-null `input_hash` and `output_hash` (and `error_json` set for failures).
  - `token_outcomes_table` has exactly one terminal outcome per token with correct outcome values.
  - `artifacts_table` records sink outputs with non-null `content_hash`/`artifact_type` and expected payload/path refs.
  - Hash integrity: recompute canonical hashes for known inputs and compare to stored hashes.
- Example pattern:
```python
from sqlalchemy import select
from elspeth.core.landscape import LandscapeDB, node_states_table, token_outcomes_table, artifacts_table

db = LandscapeDB.from_url(f"sqlite:///{landscape_db}")
with db.engine.connect() as conn:
    node_states = conn.execute(select(node_states_table)).fetchall()
    assert node_states and all(s.input_hash for s in node_states)

    outcomes = conn.execute(select(token_outcomes_table)).fetchall()
    assert outcomes and all(o.is_terminal for o in outcomes)

    artifacts = conn.execute(select(artifacts_table)).fetchall()
    assert artifacts and all(a.content_hash for a in artifacts)
```
---
# Test Defect Report

## Summary

- Tests access the private `ExecutionGraph._graph` attribute instead of using public APIs, making them brittle and coupled to internal implementation

## Severity

- Severity: minor
- Priority: P3

## Category

- Infrastructure Gaps

## Evidence

- `tests/cli/test_run_command.py:522` accesses `graph._graph.nodes()` directly.
```python
from_instances_calls.append(
    {"graph_id": id(graph), "node_ids": sorted(graph._graph.nodes())}
)
```
- `tests/cli/test_run_command.py:612` accesses `rebuilt_graph._graph.nodes()` directly.
```python
rebuilt_node_ids = set(rebuilt_graph._graph.nodes())
```
- Public alternatives exist: `ExecutionGraph.get_nx_graph()` and `ExecutionGraph.get_nodes()` in `src/elspeth/core/dag.py:68` and `src/elspeth/core/dag.py:236`.

## Impact

- Tests will fail if the internal NetworkX graph is renamed/replaced, even if behavior is unchanged.
- Discourages internal refactors and increases maintenance cost.

## Root Cause Hypothesis

- Convenience access to NetworkX internals without using available public accessors.

## Recommended Fix

- Replace `_graph` access with public methods:
```python
node_ids = {node.node_id for node in graph.get_nodes()}
# or
node_ids = set(graph.get_nx_graph().nodes())
```
- Apply the same change in both graph-reuse tests to keep them robust to internal refactors.
