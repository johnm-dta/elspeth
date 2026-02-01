# Bug Report: Quarantined source rows lack QUARANTINED token_outcome

## Summary

- Quarantined SourceRow handling creates tokens and routes them to a quarantine sink but never records a QUARANTINED token_outcome, leaving no terminal outcome for those rows.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: Unknown
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Any source emitting `SourceRow.quarantined` with a configured quarantine sink (e.g., CSV row failing schema validation)

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a source with `on_validation_failure` pointing to a valid sink.
2. Provide an input row that fails validation (source yields `SourceRow.quarantined`).
3. Run the pipeline and inspect `token_outcomes` or use `explain` for that row.

## Expected Behavior

- A QUARANTINED token_outcome is recorded (with error_hash), and lineage/explain can resolve the terminal state.

## Actual Behavior

- No token_outcome is recorded for quarantined source rows, so lineage/explain treats the row as having no terminal outcome.

## Evidence

- Quarantined branch creates token and routes to sink but does not record an outcome: `src/elspeth/engine/orchestrator.py:902`, `src/elspeth/engine/orchestrator.py:909`, `src/elspeth/engine/orchestrator.py:915`
- Outcomes are explicitly recorded for completed rows elsewhere, showing the omission is localized to the quarantine path: `src/elspeth/engine/orchestrator.py:966`
- Lineage resolution depends on token_outcomes; missing outcomes yield no terminal state: `src/elspeth/core/landscape/lineage.py:96`

## Impact

- User-facing impact: `explain`/lineage misses quarantined rows, making investigations incomplete.
- Data integrity / security impact: audit trail lacks a terminal state for quarantined rows, violating completeness.
- Performance or cost impact: None.

## Root Cause Hypothesis

- The quarantine branch in `_execute_run` never calls `recorder.record_token_outcome` for `RowOutcome.QUARANTINED` and does not hash `source_item.quarantine_error`.

## Proposed Fix

- Code changes (modules/files): Update `src/elspeth/engine/orchestrator.py` to record `RowOutcome.QUARANTINED` (with `error_hash` and `sink_name` when present) immediately after creating the quarantine token.
- Config or schema changes: None.
- Tests to add/update: Add an orchestrator test that a quarantined SourceRow yields a QUARANTINED token_outcome and is discoverable via lineage/explain.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:340`
- Observed divergence: Quarantined rows do not reach a recorded terminal state.
- Reason (if known): Missing `record_token_outcome` call in the quarantine branch.
- Alignment plan or decision needed: Record QUARANTINED outcomes at source quarantine handling.

## Acceptance Criteria

- QUARANTINED token_outcomes exist for source-quarantined rows.
- `explain` returns terminal state for quarantined rows without missing outcomes.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_row_outcome.py tests/core/landscape/test_lineage.py`
- New tests required: yes, add coverage for source quarantine outcome recording.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md:340`
---
# Bug Report: COMPLETED token_outcomes recorded before sink write

## Summary

- Orchestrator records `RowOutcome.COMPLETED` before sink writes occur; if sink.write/flush fails, token_outcomes still indicate success.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: Unknown
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Any pipeline with a sink that raises in `write()` or `flush()`

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Create a sink whose `write()` raises an exception (or `flush()` fails).
2. Run a pipeline that produces at least one completed row.
3. Inspect `token_outcomes` after the failure.

## Expected Behavior

- COMPLETED outcomes are recorded only after successful sink writes; on sink failure, outcomes remain absent or are marked FAILED.

## Actual Behavior

- COMPLETED outcomes are recorded before sink writes; they remain even if sink.write/flush fails.

## Evidence

- COMPLETED outcome recorded in the processing loop before any sink write: `src/elspeth/engine/orchestrator.py:966`
- Aggregation flush path records COMPLETED outcomes before sink write: `src/elspeth/engine/orchestrator.py:1864`, `src/elspeth/engine/orchestrator.py:1890`
- Sink writes happen later in the same method: `src/elspeth/engine/orchestrator.py:1064`
- Sink write failures raise after outcomes are already recorded: `src/elspeth/engine/executors.py:1448`
- Lineage uses token_outcomes to resolve terminal state, so these premature outcomes can mislead explain: `src/elspeth/core/landscape/lineage.py:96`

## Impact

- User-facing impact: `explain` can report rows as completed even though sink writes failed.
- Data integrity / security impact: audit trail records incorrect terminal states.
- Performance or cost impact: None.

## Root Cause Hypothesis

- `record_token_outcome` is called when a RowResult is produced rather than after sink write success, and there is no correction path on sink failure.

## Proposed Fix

- Code changes (modules/files): Move COMPLETED outcome recording in `src/elspeth/engine/orchestrator.py` to after successful sink writes (e.g., via `on_token_written` callback) and ensure failure paths record FAILED or leave no terminal outcome.
- Config or schema changes: None.
- Tests to add/update: Add an integration test that a sink write failure does not leave COMPLETED outcomes.
- Risks or migration steps: Ensure outcome recording order remains deterministic and consistent with sink batching.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:340`
- Observed divergence: COMPLETED is recorded before the output sink is successfully written.
- Reason (if known): Outcome recording placed in the per-row result loop, not in sink write completion.
- Alignment plan or decision needed: Record COMPLETED only after sink write/flush succeeds.

## Acceptance Criteria

- If a sink write fails, no COMPLETED token_outcome exists for affected tokens.
- After successful sink writes, COMPLETED outcomes are recorded with correct sink_name.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/integration/test_sink_durability.py tests/core/landscape/test_lineage.py`
- New tests required: yes, add coverage for sink failure vs token_outcome timing.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md:340`
---
# Bug Report: Plugin cleanup errors suppressed during run cleanup

## Summary

- Orchestrator suppresses exceptions from plugin `on_complete` and `close`, masking system-owned plugin bugs and allowing runs to succeed despite cleanup failures.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: Unknown
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Any pipeline with a plugin that raises in `on_complete()` or `close()`

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Implement a transform or sink that raises in `on_complete()`.
2. Run a pipeline that invokes that plugin.
3. Observe that the run completes without surfacing the exception.

## Expected Behavior

- Plugin lifecycle exceptions should fail the run (system-owned plugin bugs must crash).

## Actual Behavior

- Exceptions are suppressed/logged and the run continues or completes.

## Evidence

- Transform close exceptions are caught and only logged: `src/elspeth/engine/orchestrator.py:211`
- `on_complete` exceptions are suppressed in run cleanup: `src/elspeth/engine/orchestrator.py:1126`
- `on_complete` exceptions are suppressed in resume cleanup: `src/elspeth/engine/orchestrator.py:1694`
- Spec says plugin exceptions must crash: `CLAUDE.md:166`

## Impact

- User-facing impact: pipelines can appear successful despite plugin cleanup failures.
- Data integrity / security impact: unflushed or partially written outputs may be treated as valid.
- Performance or cost impact: possible resource leaks (open files, connections).

## Root Cause Hypothesis

- Cleanup logic intentionally suppresses exceptions for best-effort cleanup, contrary to plugin ownership rules.

## Proposed Fix

- Code changes (modules/files): Remove suppression in `src/elspeth/engine/orchestrator.py` and, if needed, aggregate cleanup errors and raise after attempting all cleanups.
- Config or schema changes: None.
- Tests to add/update: Update orchestrator cleanup tests to assert exceptions propagate.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:166`
- Observed divergence: plugin lifecycle exceptions are suppressed instead of crashing the run.
- Reason (if known): best-effort cleanup choice conflicts with plugin ownership contract.
- Alignment plan or decision needed: enforce crash-on-plugin-bug even during cleanup.

## Acceptance Criteria

- Any plugin `on_complete` or `close` exception causes the run to fail.
- Cleanup still attempts all plugins, but exceptions are surfaced.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_orchestrator_cleanup.py tests/scripts/cicd/test_no_bug_hiding.py`
- New tests required: yes, add coverage for cleanup exception propagation.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md:166`
