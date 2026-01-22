# Bug Report: Plugin gate routes missing in ExecutionGraph.from_config

## Summary

- ExecutionGraph builds all `row_plugins` as transforms and omits route resolution for plugin gates, while the engine still supports `BaseGate`, leading to MissingEdgeError or incorrect routing/audit if a plugin gate emits a route label.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: 81a0925 (fix/rc1-bug-burndown-session-2)
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `src/elspeth/core/dag.py`.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `dag.py`, orchestrator, processor, executors, tests/docs for gate handling.

## Steps To Reproduce

1. Define a plugin gate (subclass of `BaseGate`) that returns `RoutingAction.route("review")` and include it in the transforms list (programmatic `PipelineConfig.transforms` or settings `row_plugins` if supported).
2. Build the graph via `ExecutionGraph.from_config(settings)`.
3. Execute the pipeline so the gate emits the `"review"` route label.

## Expected Behavior

- The graph includes gate nodes/edges and a route resolution map for plugin gate labels, so routing events are recorded and execution continues.

## Actual Behavior

- The graph treats the gate as a transform and lacks route resolution for the gate label, causing MissingEdgeError or missing routing events.

## Evidence

- Logs or stack traces: MissingEdgeError is raised when the route label is not found in `route_resolution_map` (`src/elspeth/engine/executors.py:404-411`).
- Artifacts (paths, IDs, screenshots): Graph builder treats `row_plugins` as transforms and notes plugin gates removed (`src/elspeth/core/dag.py:275-291`).
- Minimal repro input (attach or link): Engine still allows gates in transforms (`RowPlugin = BaseTransform | BaseGate`) (`src/elspeth/engine/orchestrator.py:32-36`).

## Impact

- User-facing impact: plugin-gate pipelines fail at runtime or misrecord routing events.
- Data integrity / security impact: audit trail can miss or misattribute routing decisions.
- Performance or cost impact: wasted runs and operator time.

## Root Cause Hypothesis

- `ExecutionGraph.from_config` cannot classify `row_plugins` by type and assumes only transforms, so it never registers gate routing metadata despite engine support for `BaseGate`.

## Proposed Fix

- Code changes (modules/files): Update `src/elspeth/core/dag.py` to either (a) accept plugin metadata/instances to build gate nodes and route maps for `BaseGate`, or (b) fail fast when plugin gates are configured.
- Config or schema changes: Consider adding explicit plugin type metadata in settings to enable gate/transform distinction.
- Tests to add/update: Add a graph compilation test that includes a `BaseGate` and asserts either correct route resolution or explicit rejection.
- Risks or migration steps: Requires a product decision (support plugin gates vs. hard deprecate) and may require migration guidance.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/plugins/discovery.py:157-159` (gates are config-driven system operations, not plugins).
- Observed divergence: Engine accepts `BaseGate` in transforms while graph builder assumes config-only gates.
- Reason (if known): Partial migration from plugin gates to config gates.
- Alignment plan or decision needed: Decide on gate support strategy and enforce it in DAG compilation.

## Acceptance Criteria

- A pipeline with a plugin gate either (a) compiles with correct gate routing metadata or (b) fails early with a clear error before execution.

## Tests

- Suggested tests to run: `pytest tests/engine/test_engine_gates.py`, `pytest tests/core/test_dag.py`
- New tests required: yes (plugin-gate graph compilation/rejection)

## Notes / Links

- Related issues/PRs: `docs/bugs/pending/P2-2026-01-19-plugin-gate-graph-mismatch.md`
- Related design docs: `docs/contracts/plugin-protocol.md`
---
# Bug Report: Duplicate coalesce branch names silently overwritten in DAG mapping

## Summary

- `ExecutionGraph.from_config` overwrites `branch_to_coalesce` entries when the same branch appears in multiple coalesce configs, so forked tokens for that branch are routed to only one coalesce and the other coalesce never receives required inputs.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: 81a0925 (fix/rc1-bug-burndown-session-2)
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `src/elspeth/core/dag.py`.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `dag.py` coalesce mapping and processor branch routing.

## Steps To Reproduce

1. Configure a fork gate with `fork_to=["path_a","path_b"]`.
2. Configure two coalesce entries that both include `"path_a"` (e.g., coalesce A branches `["path_a","path_b"]`, coalesce B branches `["path_a","path_c"]`).
3. Build the graph and run a row through the fork.

## Expected Behavior

- Graph compilation rejects ambiguous coalesce branch assignments (or provides explicit fan-out semantics).

## Actual Behavior

- The later coalesce assignment overwrites the earlier mapping; one coalesce never receives `"path_a"` and cannot merge correctly.

## Evidence

- Logs or stack traces: Unknown (static analysis).
- Artifacts (paths, IDs, screenshots): `branch_to_coalesce` is overwritten without duplicate checks (`src/elspeth/core/dag.py:409-416`).
- Minimal repro input (attach or link): Forked child routing uses `_branch_to_coalesce` to select coalesce target (`src/elspeth/engine/processor.py:699-706`).

## Impact

- User-facing impact: missing or delayed coalesce outputs for one of the configured merges.
- Data integrity / security impact: audit trail shows incomplete merges or timeouts for expected coalesce paths.
- Performance or cost impact: extra retries/timeouts and wasted processing.

## Root Cause Hypothesis

- DAG builder uses a plain dict for `branch_to_coalesce` with no validation, so duplicate branch names across coalesce configs silently override prior mappings.

## Proposed Fix

- Code changes (modules/files): Add duplicate-branch detection in `src/elspeth/core/dag.py` when building `branch_to_coalesce` and raise `GraphValidationError` on conflicts.
- Config or schema changes: Optionally add a Pydantic validator in `src/elspeth/core/config.py` to enforce unique branches across coalesce configs.
- Tests to add/update: Add a DAG test that defines duplicate branch names across coalesce configs and asserts a validation error.
- Risks or migration steps: None beyond rejecting ambiguous configurations.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/core/dag.py:55` (branch_name -> coalesce_name implies a 1:1 mapping).
- Observed divergence: Duplicate branch names overwrite earlier mappings.
- Reason (if known): No duplicate validation in DAG construction.
- Alignment plan or decision needed: Enforce uniqueness or explicitly define fan-out semantics.

## Acceptance Criteria

- Duplicate branch names across coalesce configs are rejected at graph build time with a clear error message.

## Tests

- Suggested tests to run: `pytest tests/core/test_dag.py -k coalesce`
- New tests required: yes (duplicate branch validation)

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
