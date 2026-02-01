# Bug Report: Duplicate Config Gate Names Overwrite Node Mapping

## Summary

Gate names are documented as unique but not validated, so duplicates overwrite `config_gate_id_map` and cause multiple gates to share a node ID, corrupting routing/audit attribution.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (static analysis agent)
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: main (d8df733)
- OS: Linux
- Python version: 3.12+
- Config profile / env vars: Pipeline configuration with duplicate gate names
- Data set or fixture: Any

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `src/elspeth/core/config.py`
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed config.py, dag.py, processor.py

## Steps To Reproduce

1. Create a config with two gates that share the same `name`
2. Load with `load_settings()` and build the graph with `ExecutionGraph.from_config()`
3. Inspect `graph.get_config_gate_id_map()` or run the pipeline
4. Both gates resolve to the same node ID

## Expected Behavior

- Duplicate gate names are rejected at config validation with a clear error

## Actual Behavior

- Duplicate gate names are accepted
- The last gate wins in the map
- Earlier gates are misattributed

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots):
  - `src/elspeth/core/config.py:182` marks gate names as unique but no uniqueness validator exists
  - `src/elspeth/core/config.py:681` only enforces unique aggregation names (no gate-name validation)
  - `src/elspeth/core/dag.py:332` uses `gate_config.name` as the dict key, overwriting duplicates
  - `src/elspeth/engine/processor.py:885` resolves node IDs by gate name, so duplicates share a node ID
- Minimal repro input (attach or link): Config YAML with two gates having the same name

## Impact

- User-facing impact: Gate routing behavior becomes unpredictable when duplicate names are used
- Data integrity / security impact: Audit trail can attribute decisions to the wrong gate node
- Performance or cost impact: Potential reruns/debug time; otherwise minimal

## Root Cause Hypothesis

Missing uniqueness validation for `gates[*].name` in `ElspethSettings`.

## Proposed Fix

- Code changes (modules/files): Add a `model_validator` in `src/elspeth/core/config.py` to enforce unique gate names (similar to aggregation names)
- Config or schema changes: None
- Tests to add/update: Add a unit test for duplicate gate names in `tests/core/test_config.py`
- Risks or migration steps: Existing configs with duplicate gate names will fail fast instead of running incorrectly

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: Config accepts duplicate gate identifiers despite being described as unique
- Reason (if known): Unknown
- Alignment plan or decision needed: Enforce uniqueness at config validation

## Acceptance Criteria

- Configs with duplicate gate names fail validation with a clear error message
- Unique gate names continue to validate and map 1:1 to node IDs

## Tests

- Suggested tests to run: `pytest tests/core/test_config.py -k gate`
- New tests required: Yes, duplicate gate name validation

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown

## Verification Status

- [x] Bug confirmed via reproduction
- [x] Root cause verified
- [x] Fix implemented
- [x] Tests added
- [x] Fix verified

## Resolution

**Fixed in commit:** (this commit)

**Changes:**
- Added `validate_unique_gate_names` validator to `src/elspeth/core/config.py`
- Added `validate_unique_coalesce_names` validator (parallel vulnerability identified during fix)
- Added 4 tests covering duplicate name rejection for both gates and coalesce

**Date closed:** 2026-01-23
