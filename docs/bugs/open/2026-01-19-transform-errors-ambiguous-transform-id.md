# Bug Report: Transform errors are recorded under plugin `name`, not unique node ID (ambiguous when plugin is used multiple times)

## Summary

- When a transform returns `TransformResult.error()`, `TransformExecutor` records a transform error event via `ctx.record_transform_error(transform_id=transform.name, ...)`.
- Landscape stores this value in `transform_errors.transform_id`, which becomes ambiguous if a pipeline uses the same plugin multiple times (e.g., two `field_mapper` nodes with different configs).
- This undermines audit traceability and makes it difficult to map errors back to the correct DAG node.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `8cfebea78be241825dd7487fed3773d89f2d7079`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 6 (plugins), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Build a pipeline with two instances of the same transform plugin (e.g., two `field_mapper` nodes) and configure both to route errors (`on_error`) to a sink.
2. Provide input that causes one of the instances to return `TransformResult.error()`.
3. Inspect `transform_errors` in Landscape for the new record(s).
4. Observe `transform_id` contains the plugin name (e.g., `"field_mapper"`) rather than the unique node ID, making the record ambiguous.

## Expected Behavior

- Transform error records should attribute failures to the specific DAG node (node ID) that produced the error.

## Actual Behavior

- Transform error records store `transform_id=transform.name`, which is not unique per pipeline/run.

## Evidence

- TransformExecutor records errors with `transform_id=transform.name`: `src/elspeth/engine/executors.py:228-268`
- PluginContext persists `transform_id` exactly as provided: `src/elspeth/plugins/context.py:174-223`
- Landscape recorder stores `transform_id` in `transform_errors`: `src/elspeth/core/landscape/recorder.py:2099-2141`

## Impact

- User-facing impact: investigating which node failed is harder (especially in pipelines with repeated plugin types).
- Data integrity / security impact: forces inference from other data when error attribution should be explicit.
- Performance or cost impact: increased incident triage time.

## Root Cause Hypothesis

- Executor uses the transform’s human-readable `name` instead of its unique `node_id`, even though `transform.node_id` is available and asserted non-None.

## Proposed Fix

- Code changes (modules/files):
  - Prefer storing node IDs for error attribution:
    - change `TransformExecutor` to pass `transform_id=transform.node_id` (or add a dedicated `node_id` column/parameter and keep `plugin_name` separately).
  - If backwards compatibility is needed, store both: `{transform_node_id, plugin_name}`.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test pipeline with two transforms sharing `name` and assert transform error records disambiguate via node ID.
- Risks or migration steps:
  - If the DB schema column name `transform_id` is already used, consider a schema migration to rename to `node_id` or add a new column.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (“Every decision must be traceable to source data, configuration, and code version”)
- Observed divergence: transform errors are not traceable to the precise DAG node when names repeat.
- Reason (if known): early implementation used plugin names as identifiers.
- Alignment plan or decision needed: define the canonical identity used in error tables (node ID vs plugin name).

## Acceptance Criteria

- For any transform error record, it’s possible to map the record to exactly one DAG node without inference.

## Tests

- Suggested tests to run: `pytest tests/engine/ tests/core/landscape/`
- New tests required: yes

## Notes / Links

- Related docs: `docs/contracts/plugin-protocol.md` (node identity expectations)
