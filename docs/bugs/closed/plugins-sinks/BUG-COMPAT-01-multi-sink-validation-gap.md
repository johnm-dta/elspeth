# Bug Report: Resume Compatibility Ignores Non-Ancestor Branches in Multi-Sink DAGs

## Summary

- Checkpoint compatibility validation only hashes the ancestors of the checkpoint node (a sink), so changes to other sink branches are ignored, allowing resume with mixed pipeline configurations inside a single run.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Branch Bug Scan (fix/rc1-bug-burndown-session-4)
- Date: 2026-01-25
- Related run/issue ID: BUG-COMPAT-01

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Synthetic DAG with a gate routing to sink_a and sink_b

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of compatibility.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Build a graph with a gate that routes to sink_a and sink_b.
2. Run pipeline until checkpoint created for a row routed to sink_a.
3. Modify only the transform or sink config in the sink_b branch (nodes not in sink_a's ancestor set).
4. Call `CheckpointCompatibilityValidator.validate(checkpoint, modified_graph)`.

## Expected Behavior

- Resume should be rejected because the pipeline configuration changed for a branch that can still process unprocessed rows, breaking run-level configuration consistency.

## Actual Behavior

- Validator returns `can_resume=True` because it only compares the upstream topology for the checkpoint's sink branch, ignoring other branches.

## Evidence

- `src/elspeth/core/checkpoint/compatibility.py:76` computes the topology hash using `checkpoint.node_id`, scoping validation to a single node's ancestors.
- `src/elspeth/core/canonical.py:188` uses `nx.ancestors` to build the upstream subgraph, excluding other branches not leading to the checkpoint node.
- `src/elspeth/engine/orchestrator.py:150` documents checkpoints are created after sink writes and `node_id` is the sink node, so the checkpoint is tied to a single sink branch.

Example DAG:
```
Source → Transform1 → Gate → Transform2 → Sink_A (checkpointed here)
                            ↘ Transform3 → Sink_B (modified after checkpoint)
```

Current behavior: Resume allowed (only Sink_A ancestors checked)
Expected: Resume rejected (full DAG or all sink branches must match)

## Impact

- User-facing impact: Resume can proceed despite config changes on other branches, producing inconsistent outputs within the same run_id.
- Data integrity / security impact: Auditability is compromised because a single run contains decisions made under different pipeline configurations. The audit trail shows one run_id but multiple configs.
- Performance or cost impact: Potential rework and reprocessing if inconsistent results are detected during audit reviews.

## Root Cause Hypothesis

- Compatibility validation compares only the upstream topology for the checkpoint's sink node, which omits other sink branches in multi-sink DAGs.

## Proposed Fix

- Code changes (modules/files):

  **Option A: Validate full DAG topology (recommended for simplicity)**
  ```python
  # In compatibility.py
  def validate_compatibility(checkpoint, graph):
      stored_hash = checkpoint.upstream_topology_hash
      # Compute hash of ENTIRE graph, not just ancestors
      full_graph_hash = compute_full_topology_hash(graph)
      return stored_hash == full_graph_hash

  # In manager.py create_checkpoint()
  upstream_topology_hash = compute_full_topology_hash(graph)
  ```

  **Option B: Store per-sink branch hashes (more granular)**
  ```python
  # In manager.py create_checkpoint()
  sink_hashes = {}
  for sink_node in graph.get_sinks():
      sink_hashes[sink_node] = compute_upstream_topology_hash(graph, sink_node)
  checkpoint.sink_branch_hashes = json.dumps(sink_hashes)

  # In compatibility.py validate()
  stored_hashes = json.loads(checkpoint.sink_branch_hashes)
  current_hashes = {
      sink: compute_upstream_topology_hash(graph, sink)
      for sink in graph.get_sinks()
  }
  return stored_hashes == current_hashes
  ```

- Config or schema changes:
  - Option A: No schema change (reuse `upstream_topology_hash` column)
  - Option B: Add new column `sink_branch_hashes TEXT` (JSON-serialized dict)

- Tests to add/update:
  - `test_resume_rejects_parallel_branch_changes()` - Checkpoint at sink_a, modify sink_b branch, verify rejection
  - `test_resume_allows_same_config_multi_sink()` - No changes, verify resume allowed
  - `test_resume_rejects_new_sink_added()` - Add new sink after checkpoint, verify rejection

- Risks or migration steps:
  - Breaking change: Existing checkpoints without full-graph hash will be rejected
  - Acceptable per CLAUDE.md No Legacy Code Policy
  - Alembic migration required if Option B chosen (add column)

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:13` - Auditability Standard
- Observed divergence: Resume validation allows a run to continue even though parts of the pipeline configuration differ from the original run, violating "every decision must be traceable to source data, configuration, and code version."
- Reason (if known): Compatibility check is scoped to a single sink's ancestor subgraph rather than the full DAG.
- Alignment plan or decision needed: Require full-DAG topology hash in checkpoints or reject resume when other sink branches exist and cannot be validated.

## Acceptance Criteria

- Resume is rejected if any non-ancestor branch (e.g., another sink path) changes after the checkpoint.
- A new test covering multi-sink branch changes fails on current behavior and passes after the fix.
- Audit trail integrity: Single run_id never contains outputs from different pipeline configurations.

## Tests

- Suggested tests to run: `pytest tests/core/checkpoint/test_compatibility_validator.py`
- New tests required: yes, add multi-sink branch change detection case (3 tests listed above)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs:
  - `docs/bugs/BRANCH_BUG_TRIAGE_2026-01-25.md` - Bug triage report
  - `CLAUDE.md:13` - Auditability Standard
  - `src/elspeth/core/checkpoint/compatibility.py` - Validation logic
