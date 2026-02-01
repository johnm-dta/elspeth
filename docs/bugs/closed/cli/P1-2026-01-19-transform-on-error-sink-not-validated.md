# Bug Report: Transform `on_error` sink destination is not validated; invalid sink can crash mid-run (KeyError)

## Summary

- Transforms can return `TransformResult.error()` and request routing to an error sink via `_on_error` (configured by `TransformDataConfig.on_error`).
- `RowProcessor` converts transform errors into `RowOutcome.ROUTED` with `sink_name=error_sink`.
- `Orchestrator` assumes routed sink names exist in `PipelineConfig.sinks` and blindly indexes `pending_tokens[result.sink_name]`, which can raise `KeyError` mid-run when `on_error` references an unknown sink.

Note: source quarantine destination validation is already tracked separately in `docs/bugs/open/2026-01-19-source-quarantine-silent-drop.md`.

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

1. Configure a transform that can return `TransformResult.error()` and set `on_error: "missing_sink"`.
2. Run a pipeline where `PipelineConfig.sinks` does not contain `"missing_sink"`.
3. Trigger a transform error and observe a `KeyError` when the orchestrator buffers tokens for sinks.

## Expected Behavior

- Pipeline initialization should fail fast if a transform config sets `on_error` to a sink name that is neither `"discard"` nor present in `PipelineConfig.sinks`.

## Actual Behavior

- Missing transform error sink can crash mid-run via `KeyError`.

## Evidence

- Config allows `on_error` but only validates non-empty: `src/elspeth/plugins/config_base.py:153-174`
- Transform errors are routed via `transform._on_error`: `src/elspeth/engine/executors.py:238-268`
- RowProcessor emits `RowOutcome.ROUTED` with `sink_name=error_sink`: `src/elspeth/engine/processor.py:639-661`
- Orchestrator indexes routed sink name without validating presence: `src/elspeth/engine/orchestrator.py:653-658`

## Impact

- User-facing impact: configuration errors surface late as runtime crashes (after partial processing).
- Data integrity / security impact: partial processing can occur before failure, producing incomplete outputs and partial audit records for a run.
- Performance or cost impact: wasted time debugging and rerunning pipelines.

## Root Cause Hypothesis

- Route destination validation is applied to gate routing only; transform error-routing destinations are not validated during pipeline initialization.

## Proposed Fix

- Code changes (modules/files):
  - At pipeline initialization, validate that any transform with `_on_error` set has:
    - `_on_error == "discard"`, or
    - `_on_error in PipelineConfig.sinks`.
  - Fail fast with a clear `RouteValidationError`-style message (similar to gate route validation).
- Config or schema changes: none.
- Tests to add/update:
  - Add an orchestrator integration test where a transform errors with `on_error` pointing to an unknown sink and assert the run fails before processing rows (startup validation).
- Risks or migration steps:
  - This is a behavior tightening; treat as correct because running with invalid error routes is unsafe.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (auditability and “crash on our bugs”)
- Observed divergence: invalid error-routing configuration is not detected until runtime.
- Reason (if known): validation coverage limited to gate routes.
- Alignment plan or decision needed: decide where validation belongs (config resolution vs orchestrator run start).

## Acceptance Criteria

- Any run with invalid transform `on_error` sink name fails before processing rows.

## Tests

- Suggested tests to run: `pytest tests/plugins/ tests/engine/`
- New tests required: yes

## Notes / Links

- Related ticket: `docs/bugs/open/2026-01-19-source-quarantine-silent-drop.md` (source quarantine destination validation)
