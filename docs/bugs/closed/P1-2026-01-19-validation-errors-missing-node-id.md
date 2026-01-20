# Bug Report: Validation errors are recorded without a `node_id` (PluginContext `node_id` is never set)

## Summary

- Sources call `ctx.record_validation_error(...)` on schema failures, but `PluginContext.record_validation_error()` forwards `node_id=self.node_id` to Landscape.
- `Orchestrator` constructs a single `PluginContext` and never sets `ctx.node_id` before invoking `source.load(ctx)`, so `validation_errors.node_id` is typically `NULL` and `ValidationErrorToken.node_id` falls back to `"unknown"`.
- This weakens auditability: validation errors can’t be reliably attributed to the specific source node in the run DAG.

## Severity

- Severity: major
- Priority: P1

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

1. Configure a source with a strict schema and set `on_validation_failure` to a non-`"discard"` sink.
2. Provide input containing at least one invalid row (schema validation failure).
3. Run the pipeline.
4. Inspect the Landscape DB `validation_errors` table and observe `node_id` is `NULL` (or otherwise not the actual source node ID).

## Expected Behavior

- `validation_errors.node_id` should be the source node ID from the run DAG, enabling unambiguous attribution of validation failures.

## Actual Behavior

- Validation errors are recorded with `node_id=None` because `PluginContext.node_id` is never populated.

## Evidence

- `PluginContext.record_validation_error()` forwards `node_id=self.node_id`: `src/elspeth/plugins/context.py:119-172`
- Sources call `ctx.record_validation_error(...)` on `ValidationError`: `src/elspeth/plugins/sources/csv_source.py:110-132`, `src/elspeth/plugins/sources/json_source.py:143-165`
- `Orchestrator` constructs `PluginContext` without setting `node_id` and passes it into `source.load(ctx)`: `src/elspeth/engine/orchestrator.py:538-616`
- Landscape recorder accepts `node_id: str | None` and persists it into `validation_errors.node_id`: `src/elspeth/core/landscape/recorder.py:2052-2095`

## Impact

- User-facing impact: “Which source node rejected this row?” is harder to answer (especially when multiple sources or multi-run analysis exists).
- Data integrity / security impact: audit record is less attributable; increases the need for inference, which is explicitly discouraged in `CLAUDE.md`.
- Performance or cost impact: increased investigation time for quarantine/validation issues.

## Root Cause Hypothesis

- The engine never scopes `PluginContext` to a specific plugin invocation (source/transform/sink), so `ctx.node_id` remains unset even though plugin instances have `node_id` populated.

## Proposed Fix

- Code changes (modules/files):
  - Option A (recommended): introduce a small “scoped context” helper (context manager or copy) that sets `ctx.node_id` / `ctx.plugin_name` before invoking each plugin hook (`source.load`, `transform.process`, `gate.evaluate`, `sink.write`) and restores afterward.
  - Option B: change `PluginContext.record_validation_error(...)` to accept an explicit `node_id` override, and have sources pass `self.node_id` (which the orchestrator assigns).
- Config or schema changes: none.
- Tests to add/update:
  - Add an integration test that runs a pipeline with an invalid source row and asserts the persisted `validation_errors.node_id` equals the source node ID from the graph.
- Risks or migration steps:
  - If existing DBs contain `NULL` node_id values, ensure queries tolerate it; the fix should only affect new runs.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (“No inference — if it’s not recorded, it didn’t happen”)
- Observed divergence: validation errors are recorded without the node attribution needed for end-to-end traceability.
- Reason (if known): `PluginContext` metadata fields exist but are not wired by the engine.
- Alignment plan or decision needed: decide whether `PluginContext` is a per-plugin scoped object or a global run-scoped object with explicit parameters for attribution.

## Acceptance Criteria

- Validation errors recorded during `source.load(...)` include the correct source `node_id`.
- No “unknown” node IDs are produced in `ValidationErrorToken` when running with Landscape enabled.

## Tests

- Suggested tests to run: `pytest tests/plugins/sources/ tests/core/landscape/`
- New tests required: yes

## Notes / Links

- Related design docs: `docs/design/architecture.md` (audit trail attribution expectations)
