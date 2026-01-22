# Bug Report: Duplicate Config Gate Names Overwrite Node Mapping

## Summary

- Gate names are documented as unique but not validated, so duplicates overwrite `config_gate_id_map` and cause multiple gates to share a node ID, corrupting routing/audit attribution.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `src/elspeth/core/config.py`
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: reviewed `src/elspeth/core/config.py`, `src/elspeth/core/dag.py`, `src/elspeth/engine/processor.py`

## Steps To Reproduce

1. Create a config with two gates that share the same `name`.
2. Load with `load_settings()` and build the graph with `ExecutionGraph.from_config()`.
3. Inspect `graph.get_config_gate_id_map()` or run the pipeline; both gates resolve to the same node ID.

## Expected Behavior

- Duplicate gate names are rejected at config validation with a clear error.

## Actual Behavior

- Duplicate gate names are accepted; the last gate wins in the map, and earlier gates are misattributed.

## Evidence

- `src/elspeth/core/config.py:182` marks gate names as unique but no uniqueness validator exists.
- `src/elspeth/core/config.py:681` only enforces unique aggregation names (no gate-name validation).
- `src/elspeth/core/dag.py:332` uses `gate_config.name` as the dict key, overwriting duplicates.
- `src/elspeth/engine/processor.py:885` resolves node IDs by gate name, so duplicates share a node ID.

## Impact

- User-facing impact: Gate routing behavior becomes unpredictable when duplicate names are used.
- Data integrity / security impact: Audit trail can attribute decisions to the wrong gate node.
- Performance or cost impact: Potential reruns/debug time; otherwise minimal.

## Root Cause Hypothesis

- Missing uniqueness validation for `gates[*].name` in `ElspethSettings`.

## Proposed Fix

- Code changes (modules/files): Add a `model_validator` in `src/elspeth/core/config.py` to enforce unique gate names (similar to aggregation names).
- Config or schema changes: None.
- Tests to add/update: Add a unit test for duplicate gate names in `tests/core/test_config.py`.
- Risks or migration steps: Existing configs with duplicate gate names will fail fast instead of running incorrectly.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: Config accepts duplicate gate identifiers despite being described as unique.
- Reason (if known): Unknown
- Alignment plan or decision needed: Enforce uniqueness at config validation.

## Acceptance Criteria

- Configs with duplicate gate names fail validation with a clear error message.
- Unique gate names continue to validate and map 1:1 to node IDs.

## Tests

- Suggested tests to run: `pytest tests/core/test_config.py -k gate`
- New tests required: Yes, duplicate gate name validation.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
---
# Bug Report: Duplicate Fork/Coalesce Branch Names Break Merge Semantics

## Summary

- `fork_to` and `coalesce.branches` allow duplicate branch names; coalesce tracking uses a dict keyed by branch name, so duplicates overwrite tokens and can prevent `require_all/quorum` merges from ever completing.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `src/elspeth/core/config.py`
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: reviewed `src/elspeth/core/config.py`, `src/elspeth/core/dag.py`, `src/elspeth/engine/coalesce_executor.py`

## Steps To Reproduce

1. Define a gate with `routes: {all: fork}` and `fork_to: ["path_a", "path_a"]` (duplicate branch).
2. Define a coalesce with `branches: ["path_a", "path_a"]` and `policy: require_all`.
3. Run the pipeline; coalesce never completes because only one unique branch can arrive.

## Expected Behavior

- Duplicate branch names are rejected at config validation for both `fork_to` and `coalesce.branches`.

## Actual Behavior

- Duplicates are accepted; coalesce overwrites arrivals and may stall indefinitely or drop a token.

## Evidence

- `src/elspeth/core/config.py:238` validates `fork_to` only for reserved labels, not uniqueness.
- `src/elspeth/core/config.py:327` defines `CoalesceSettings.branches` without uniqueness validation.
- `src/elspeth/engine/coalesce_executor.py:172` stores arrivals by `branch_name`, overwriting duplicates.
- `src/elspeth/engine/coalesce_executor.py:195` compares `arrived_count` to `len(settings.branches)`, so duplicates can prevent merges.
- `src/elspeth/core/dag.py:415` maps `branch_to_coalesce` by branch name, overwriting duplicates across coalesce configs.

## Impact

- User-facing impact: Pipelines can hang at coalesce or route fewer results than expected.
- Data integrity / security impact: Tokens can be overwritten, causing silent loss of branch results.
- Performance or cost impact: Runs may stall until timeout or require manual intervention.

## Root Cause Hypothesis

- Missing uniqueness checks for branch lists (`fork_to`, `coalesce.branches`) and across coalesce configurations.

## Proposed Fix

- Code changes (modules/files): Add uniqueness validation in `GateSettings.validate_fork_to_labels` and a new validator in `CoalesceSettings` (or `ElspethSettings`) to enforce unique branch names; optionally validate global uniqueness across coalesce definitions.
- Config or schema changes: None.
- Tests to add/update: Add tests for duplicate `fork_to` and duplicate `coalesce.branches` in `tests/core/test_config.py`.
- Risks or migration steps: Configs with duplicate branch names will fail fast.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: Branch identifiers are treated as unique keys at runtime but not enforced in config.
- Reason (if known): Unknown
- Alignment plan or decision needed: Enforce uniqueness at config validation.

## Acceptance Criteria

- Duplicate branch names in `fork_to` or `coalesce.branches` are rejected with a clear error.
- Coalesce merges complete when all distinct branches arrive.

## Tests

- Suggested tests to run: `pytest tests/core/test_config.py -k coalesce`
- New tests required: Yes, duplicate branch validation.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
