# Bug Report: Forked branches never coalesce because branch_to_coalesce uses node IDs

## Summary

- ExecutionGraph.from_plugin_instances stores branch_to_coalesce values as coalesce node IDs, but RowProcessor and coalesce_step_map expect coalesce names, so forked tokens never reach the coalesce executor.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 86357898ee109a1dbb8d60f3dc687983fa22c1f0
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: N/A
- Data set or fixture: Fork gate + coalesce settings (e.g., fork_to=["path_a","path_b"], coalesce branches=["path_a","path_b"])

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of dag.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a pipeline with a fork gate (fork_to paths) and a coalesce config for those branches, then build the graph via ExecutionGraph.from_plugin_instances.
2. Run the pipeline through Orchestrator and observe forked tokens after the gate.

## Expected Behavior

- Forked tokens should be held and merged at the configured coalesce point, producing a COALESCED outcome.

## Actual Behavior

- Forked tokens bypass coalescing because coalesce_name lookup fails (branch_to_coalesce stores node IDs), so coalesce_at_step remains None and coalesce executor is never called.

## Evidence

- `src/elspeth/core/dag.py:505` creates `cid` as the coalesce node ID.
- `src/elspeth/core/dag.py:510` assigns `branch_to_coalesce[branch_name] = cid` (node ID, not name).
- `src/elspeth/engine/processor.py:116` documents branch_to_coalesce as branch_name -> coalesce_name.
- `src/elspeth/engine/processor.py:694` reads branch_to_coalesce into `child_coalesce_name`, and `src/elspeth/engine/processor.py:696` uses that key against coalesce_step_map.
- `src/elspeth/engine/orchestrator.py:815` builds coalesce_step_map keyed by coalesce name (cs.name).

## Impact

- User-facing impact: fork/coalesce pipelines emit unmerged outputs or miss expected merged outputs.
- Data integrity / security impact: audit trail lacks COALESCED terminal states for forked tokens, breaking lineage expectations.
- Performance or cost impact: extra sink writes and downstream processing for unmerged branches.

## Root Cause Hypothesis

- branch_to_coalesce stores coalesce node IDs instead of coalesce names, which do not match the keys used by coalesce_step_map and coalesce executor registration.

## Proposed Fix

- Code changes (modules/files):
  - Update `src/elspeth/core/dag.py` to store `branch_to_coalesce[branch_name] = coalesce_config.name` and resolve node IDs via `coalesce_ids[coalesce_name]` when adding edges.
  - Adjust any tests that currently expect branch_to_coalesce to contain node IDs.
- Config or schema changes: None.
- Tests to add/update:
  - Update `tests/core/test_dag.py` expectations for branch_to_coalesce.
  - Add/adjust integration test to ensure coalesce is invoked when using ExecutionGraph.from_plugin_instances.
- Risks or migration steps:
  - Ensure all consumers of branch_to_coalesce expect coalesce names after the change.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/engine/processor.py:116`
- Observed divergence: DAG builder stores node IDs in branch_to_coalesce instead of coalesce names.
- Reason (if known): Mapping uses `cid` directly during graph construction.
- Alignment plan or decision needed: Store coalesce names in branch_to_coalesce and use coalesce_id_map only for edge creation.

## Acceptance Criteria

- branch_to_coalesce maps branch_name -> coalesce_name.
- Forked tokens set coalesce_at_step and reach coalesce executor; COALESCED outcomes recorded.
- Coalesce integration tests pass with ExecutionGraph.from_plugin_instances.

## Tests

- Suggested tests to run: `pytest tests/core/test_dag.py::TestCoalesceNodes`, `pytest tests/engine/test_coalesce_integration.py::TestForkCoalescePipeline`
- New tests required: yes, verify branch_to_coalesce uses names and coalesce execution occurs

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/plans/completed/2026-01-20-coalesce-integration.md`
---
# Bug Report: Duplicate branch names across coalesce configs silently override mapping

## Summary

- ExecutionGraph.from_plugin_instances builds branch_to_coalesce without detecting duplicate branch names, so later coalesce configs overwrite earlier mappings and routes silently drift to the last coalesce.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 86357898ee109a1dbb8d60f3dc687983fa22c1f0
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: N/A
- Data set or fixture: Two coalesce configs sharing a branch name (e.g., both include "path_a")

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of dag.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Define two CoalesceSettings with overlapping branch names (e.g., both include "path_a") and build the graph via ExecutionGraph.from_plugin_instances.
2. Inspect branch_to_coalesce or run a fork pipeline and observe routing for the shared branch.

## Expected Behavior

- Graph validation should reject duplicate branch names across coalesce configs with a clear error.

## Actual Behavior

- branch_to_coalesce is silently overwritten by the last coalesce definition; the earlier coalesce never receives that branch.

## Evidence

- `src/elspeth/core/dag.py:508` iterates branches for each coalesce config.
- `src/elspeth/core/dag.py:510` assigns `branch_to_coalesce[branch_name] = ...` without checking if the key already exists.
- `src/elspeth/engine/processor.py:116` documents branch_to_coalesce as a one-to-one mapping (branch_name -> coalesce_name).

## Impact

- User-facing impact: coalesce outputs are missing or incomplete when duplicate branch names are configured.
- Data integrity / security impact: coalesce may hold tokens indefinitely or merge with missing branches, producing incorrect audit outcomes.
- Performance or cost impact: increased buffering and timeout handling for branches that never arrive.

## Root Cause Hypothesis

- Missing duplicate-branch validation in DAG construction causes silent overwrites in branch_to_coalesce.

## Proposed Fix

- Code changes (modules/files):
  - Add a duplicate-branch check in `src/elspeth/core/dag.py` when building branch_to_coalesce and raise GraphValidationError with both coalesce names.
- Config or schema changes: None.
- Tests to add/update:
  - Add a test that defines duplicate branch names across coalesce configs and expects GraphValidationError.
- Risks or migration steps:
  - Existing configs relying on ambiguous duplicate branches will now fail fast.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/engine/processor.py:116`
- Observed divergence: branch_name -> coalesce_name mapping is treated as many-to-one without validation, allowing ambiguity.
- Reason (if known): No duplicate detection during DAG build.
- Alignment plan or decision needed: Enforce uniqueness of branch names across coalesce configs at graph construction time.

## Acceptance Criteria

- Graph construction raises GraphValidationError when a branch name appears in multiple coalesce configs.
- Unique branch configurations continue to build successfully.

## Tests

- Suggested tests to run: `pytest tests/core/test_dag.py::TestCoalesceNodes`
- New tests required: yes, add a duplicate-branch validation test

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/plans/completed/2026-01-20-coalesce-integration.md`
---
# Bug Report: DAG builder masks missing plugin schema/config attributes via getattr defaults

## Summary

- ExecutionGraph.from_plugin_instances uses getattr defaults for plugin config and schema attributes, which can silently bypass required schema contracts and violates the no-defensive-programming rule for system-owned plugins.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 86357898ee109a1dbb8d60f3dc687983fa22c1f0
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: N/A
- Data set or fixture: Any plugin instance missing input_schema/output_schema or config attributes

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of dag.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Define a source/transform/sink plugin missing its required schema attribute (e.g., output_schema or input_schema).
2. Build the graph with ExecutionGraph.from_plugin_instances and observe that it succeeds with None schemas.

## Expected Behavior

- Missing required plugin attributes should raise immediately (AttributeError or GraphValidationError), preventing the pipeline from running with undefined schemas.

## Actual Behavior

- getattr defaults treat missing config/schema attributes as {} or None, causing schema validation to be skipped and the pipeline to proceed.

## Evidence

- `src/elspeth/core/dag.py:365` uses `getattr(source, "config", {})`.
- `src/elspeth/core/dag.py:372` uses `getattr(source, "output_schema", None)`.
- `src/elspeth/core/dag.py:386` uses `getattr(sink, "input_schema", None)`.
- `src/elspeth/core/dag.py:407` uses `getattr(transform, "input_schema", None)` (and output_schema in the same block).
- `src/elspeth/plugins/protocols.py:65` requires SourceProtocol.output_schema.
- `src/elspeth/plugins/protocols.py:145` requires TransformProtocol.input_schema/output_schema.
- `src/elspeth/plugins/protocols.py:405` requires SinkProtocol.input_schema.
- `CLAUDE.md:492` prohibits getattr defaults for system-owned code to avoid bug-hiding.

## Impact

- User-facing impact: pipelines may run with missing or mismatched schemas without failing fast.
- Data integrity / security impact: schema compatibility checks are bypassed, risking incorrect audit trails and downstream processing errors.
- Performance or cost impact: wasted processing on invalid pipelines.

## Root Cause Hypothesis

- Defensive getattr defaults mask missing required plugin attributes instead of enforcing schema contracts.

## Proposed Fix

- Code changes (modules/files):
  - Replace getattr defaults with direct attribute access in `src/elspeth/core/dag.py`.
  - Raise GraphValidationError when required schema attributes are None or missing.
- Config or schema changes: None.
- Tests to add/update:
  - Add a test plugin missing schema attributes and assert graph construction fails fast.
- Risks or migration steps:
  - Any nonconforming plugins will now surface their contract violations immediately.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:492`
- Observed divergence: DAG builder uses getattr defaults, masking missing attributes in system-owned plugins.
- Reason (if known): Defensive defaults were used during attribute extraction.
- Alignment plan or decision needed: Enforce direct attribute access and explicit validation for required schema/config fields.

## Acceptance Criteria

- Missing plugin schema attributes cause graph construction to fail immediately.
- Schema compatibility validation is not bypassed by None defaults.

## Tests

- Suggested tests to run: `pytest tests/core/test_dag.py::TestExecutionGraph`
- New tests required: yes, add a negative test for missing schema attributes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
