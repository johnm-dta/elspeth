# Bug Report: run validates one graph but executes another

## Summary

- `run` validates ExecutionGraph instance A, then `_execute_pipeline` rebuilds a new graph with UUID-based node IDs and executes it without `validate()`, so the validated graph is discarded and the executed graph is not explicitly validated.

## Severity

- Severity: minor
- Priority: P2

## Reporter

- Name or handle: codex-cli
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of `/home/john/elspeth-rapid/src/elspeth/cli.py`.
- Model/version: GPT-5 Codex
- Tooling and permissions (sandbox/approvals): Read-only filesystem sandbox, approvals disabled.
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `src/elspeth/cli.py`, `src/elspeth/core/dag.py`, `src/elspeth/engine/orchestrator.py`.

## Steps To Reproduce

1. Create a minimal valid `settings.yaml` with at least one source and sink.
2. In a Python shell, call `ExecutionGraph.from_config(config)` twice and compare node IDs; the IDs differ because they are UUID-based.
3. Run `elspeth run -s settings.yaml --execute` and observe (by code inspection or instrumented logs) that the validated graph is not the one passed into `_execute_pipeline`.

## Expected Behavior

- The execution path uses the same validated graph instance, or the graph instance actually used for execution is validated.

## Actual Behavior

- `run` validates one graph instance and `_execute_pipeline` executes a new, unvalidated graph instance with different node IDs.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/cli.py:147`, `src/elspeth/cli.py:149`, `src/elspeth/cli.py:325`, `src/elspeth/cli.py:326`, `src/elspeth/core/dag.py:244`, `src/elspeth/core/dag.py:248`, `src/elspeth/engine/orchestrator.py:396`, `src/elspeth/engine/orchestrator.py:407`
- Minimal repro input (attach or link): Unknown

## Impact

- User-facing impact: Validation output can be misleading because it refers to a different graph instance than the one executed.
- Data integrity / security impact: Low today, but it weakens the guarantee that only validated graphs run; future validation logic changes could allow invalid graphs to execute.
- Performance or cost impact: Minimal (double graph build).

## Root Cause Hypothesis

- `_execute_pipeline` reconstructs an `ExecutionGraph` instead of reusing the validated instance, and it does not call `graph.validate()` on the new graph.

## Proposed Fix

- Code changes (modules/files): Update `src/elspeth/cli.py` to pass the validated `ExecutionGraph` into `_execute_pipeline`, or move graph validation inside `_execute_pipeline` and remove the earlier build.
- Config or schema changes: None.
- Tests to add/update: Add a test that asserts only one `ExecutionGraph.from_config` call is made during `elspeth run`, or that the instance passed to the orchestrator is validated.
- Risks or migration steps: Low; only affects CLI execution flow.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/engine/orchestrator.py:405` (expects a pre-validated graph).
- Observed divergence: CLI executes a different, unvalidated graph instance.
- Reason (if known): Unknown.
- Alignment plan or decision needed: Pass the validated graph instance through to execution.

## Acceptance Criteria

- `elspeth run` builds a single graph instance, validates it, and executes that same instance (or validates the instance used for execution).
- A unit test fails if `_execute_pipeline` rebuilds the graph.

## Tests

- Suggested tests to run: `python -m pytest tests/` (or the repository's CLI test suite).
- New tests required: Unit test to ensure `_execute_pipeline` uses the validated graph instance.

## Notes / Links

- Related issues/PRs: `docs/bugs/open/P2-2026-01-20-cli-run-rebuilds-unvalidated-graph.md`
- Related design docs: Unknown
---
# Bug Report: resume uses new aggregation node IDs that don’t match stored graph

## Summary

- `_build_resume_pipeline_config` derives aggregation node IDs from a freshly built `ExecutionGraph`, but `resume` executes with a graph reconstructed from the database; because node IDs are UUID-based, aggregation transforms and `aggregation_settings` are keyed to IDs that do not exist in the DB graph, breaking resume for pipelines with aggregations.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex-cli
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of `/home/john/elspeth-rapid/src/elspeth/cli.py`.
- Model/version: GPT-5 Codex
- Tooling and permissions (sandbox/approvals): Read-only filesystem sandbox, approvals disabled.
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `src/elspeth/cli.py`, `src/elspeth/core/dag.py`, `src/elspeth/engine/orchestrator.py`.

## Steps To Reproduce

1. Configure a pipeline with an aggregation transform in `settings.yaml`.
2. Run a pipeline and ensure it produces a run with checkpoints in the Landscape DB.
3. Invoke `elspeth resume <run_id> --execute`.
4. Observe resume fails or misbehaves when hitting aggregation nodes (e.g., foreign key errors on node IDs or aggregation treated as a normal transform).

## Expected Behavior

- Resume uses the same aggregation node IDs as the original run, so aggregation transforms are recognized and recorded under existing nodes.

## Actual Behavior

- Resume uses aggregation node IDs from a new graph instance, which do not match the IDs stored in the DB graph.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/cli.py:695`, `src/elspeth/cli.py:698`, `src/elspeth/cli.py:713`, `src/elspeth/cli.py:715`, `src/elspeth/cli.py:913`, `src/elspeth/cli.py:915`, `src/elspeth/core/dag.py:244`, `src/elspeth/core/dag.py:248`, `src/elspeth/engine/orchestrator.py:1094`
- Minimal repro input (attach or link): Unknown

## Impact

- User-facing impact: Resume fails for pipelines that include aggregation transforms.
- Data integrity / security impact: High risk of audit corruption or foreign key failures if aggregation node IDs do not exist in the run’s node table.
- Performance or cost impact: Wasted operator time; reruns from scratch may be required.

## Root Cause Hypothesis

- `_build_resume_pipeline_config` builds a fresh `ExecutionGraph` to map aggregation node IDs, but resume uses a different graph reconstructed from DB, and node IDs are random per graph build.

## Proposed Fix

- Code changes (modules/files): Update `src/elspeth/cli.py` so `_build_resume_pipeline_config` accepts the DB graph (or its aggregation ID map) and uses that for `aggregation_settings` and `transform.node_id`. Avoid constructing a new graph for resume.
- Config or schema changes: None.
- Tests to add/update: Add a resume test with an aggregation pipeline that asserts aggregation node IDs match the DB graph and resume completes.
- Risks or migration steps: Low; only affects resume graph wiring.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/engine/orchestrator.py:1094` (resume expects the same ExecutionGraph used for the original run).
- Observed divergence: CLI builds a new graph for aggregation IDs while using the DB graph for execution.
- Reason (if known): Unknown.
- Alignment plan or decision needed: Use the DB graph’s aggregation IDs when building the resume PipelineConfig.

## Acceptance Criteria

- Resuming a run with aggregations uses node IDs that exist in the original run’s graph.
- Resume completes without node ID foreign key errors or aggregation misclassification.

## Tests

- Suggested tests to run: `python -m pytest tests/` (or resume-related tests).
- New tests required: Resume integration test for aggregation pipelines.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
---
# Bug Report: resume forces `mode=append` on all sinks, breaking JSON/Database sinks

## Summary

- `_build_resume_pipeline_config` unconditionally injects `sink_options["mode"] = "append"` for every sink; plugin configs forbid unknown fields, and JSON/Database sinks do not accept `mode`, so resume fails with configuration validation errors when those sinks are present.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex-cli
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of `/home/john/elspeth-rapid/src/elspeth/cli.py`.
- Model/version: GPT-5 Codex
- Tooling and permissions (sandbox/approvals): Read-only filesystem sandbox, approvals disabled.
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed `src/elspeth/cli.py`, `src/elspeth/plugins/config_base.py`, sink configs.

## Steps To Reproduce

1. Configure a pipeline with a JSON sink or Database sink in `settings.yaml`.
2. Run the pipeline to create a run with checkpoints.
3. Execute `elspeth resume <run_id> --execute`.
4. Observe resume fails with a plugin config validation error due to the unexpected `mode` field.

## Expected Behavior

- Resume should only set options that are valid for the sink type, or use sink-specific append semantics (e.g., `if_exists=append` for database sinks).

## Actual Behavior

- Resume adds `mode=append` to all sinks, causing config validation errors for sinks that do not define `mode`.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/cli.py:718`, `src/elspeth/cli.py:724`, `src/elspeth/plugins/config_base.py:42`, `src/elspeth/plugins/sinks/json_sink.py:21`, `src/elspeth/plugins/sinks/json_sink.py:27`, `src/elspeth/plugins/sinks/database_sink.py:34`, `src/elspeth/plugins/sinks/database_sink.py:42`
- Minimal repro input (attach or link): Unknown

## Impact

- User-facing impact: Resume fails for pipelines that use JSON or Database sinks.
- Data integrity / security impact: Prevents recovery; no direct corruption, but blocks audit continuation.
- Performance or cost impact: Manual intervention required; potential reruns.

## Root Cause Hypothesis

- CLI enforces a CSV-style `mode=append` for all sinks, but sink configs are strict (`extra="forbid"`) and some sinks do not support that option.

## Proposed Fix

- Code changes (modules/files): In `src/elspeth/cli.py`, only set `mode` for sinks that support it (e.g., CSV). For database sinks, set `if_exists="append"`; for JSON sinks, either error clearly or add explicit resume-safe handling (e.g., JSONL append).
- Config or schema changes: None.
- Tests to add/update: Add a resume test with JSON and Database sinks to ensure no config validation errors.
- Risks or migration steps: Low; adjust CLI options per sink type.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/plugins/config_base.py:42` (plugin configs forbid unknown fields).
- Observed divergence: CLI injects an option not supported by all sink configs.
- Reason (if known): Unknown.
- Alignment plan or decision needed: Apply sink-specific append semantics.

## Acceptance Criteria

- Resume works with JSON and Database sinks without config validation errors.
- CSV sinks still use append mode during resume.

## Tests

- Suggested tests to run: `python -m pytest tests/` (or resume-related tests).
- New tests required: Resume tests for JSON and Database sinks.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
