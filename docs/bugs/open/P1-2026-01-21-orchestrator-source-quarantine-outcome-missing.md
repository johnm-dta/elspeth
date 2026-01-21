# Bug Report: Source quarantined rows are routed without QUARANTINED outcome

## Summary

- When `SourceRow.is_quarantined` is true, the orchestrator creates a token and routes it to the quarantine sink but never records a `RowOutcome.QUARANTINED`.
- The `quarantine_error` payload on `SourceRow` is ignored, so quarantined rows appear as completed sink outputs or lack a terminal outcome in the audit trail.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: any source that yields `SourceRow.quarantined(...)`

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a source with schema validation and `on_validation_failure` pointing to a quarantine sink.
2. Supply at least one invalid row so the source yields `SourceRow.quarantined(...)`.
3. Run the pipeline and inspect `token_outcomes` for the quarantined token.

## Expected Behavior

- Quarantined rows should record `RowOutcome.QUARANTINED` with an error hash derived from `SourceRow.quarantine_error`, while still routing to the quarantine sink for storage.

## Actual Behavior

- The orchestrator routes the quarantined row to a sink without calling `record_token_outcome`, so the audit trail lacks a QUARANTINED terminal outcome (or shows only a completed sink node_state).

## Evidence

- Quarantined rows are routed directly to sinks without recording outcomes in `src/elspeth/engine/orchestrator.py:780-795`.
- The standard QUARANTINED outcome path records `record_token_outcome` in `src/elspeth/engine/processor.py:779-790`, but this path is bypassed.
- `SourceRow.quarantine_error` exists but is unused in this flow (`src/elspeth/contracts/results.py:283-322`).

## Impact

- User-facing impact: quarantine metrics and audit queries cannot reliably identify quarantined rows.
- Data integrity / security impact: terminal state guarantees are violated; audit trail misrepresents failure handling.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Source quarantine handling bypasses RowProcessor logic and does not record a terminal outcome or validation error for the quarantined token.

## Proposed Fix

- Code changes (modules/files):
  - In `src/elspeth/engine/orchestrator.py`, when `source_item.is_quarantined` is true, compute an error_hash from `source_item.quarantine_error` and call `record_token_outcome(..., outcome=RowOutcome.QUARANTINED, error_hash=...)` before routing to the quarantine sink.
  - Optionally record validation error details using `PluginContext.record_validation_error()` if available.
- Config or schema changes: N/A
- Tests to add/update:
  - Source quarantine flow records QUARANTINED outcome and error hash.
- Risks or migration steps:
  - Ensure QUARANTINED outcomes remain terminal even when a quarantine sink is used.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md "Terminal Row States" and "Auditability Standard".
- Observed divergence: quarantined rows do not reach a terminal QUARANTINED outcome.
- Reason (if known): quarantine handling is in orchestrator without outcome recording.
- Alignment plan or decision needed: define required audit records for source quarantine flows.

## Acceptance Criteria

- Every quarantined source row records a QUARANTINED outcome with an error hash.
- Audit queries can distinguish quarantined rows from completed rows written to normal sinks.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k quarantine -v`
- New tests required: yes, source quarantine outcome recording.

## Notes / Links

- Related issues/PRs: P1-2026-01-21-validation-error-recording-crashes-on-nondict
- Related design docs: CLAUDE.md auditability standard
