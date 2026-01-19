# Bug Report: Quarantined SourceRow can be silently dropped if quarantine_destination is missing/unknown

## Summary

- When a source yields `SourceRow.quarantined(...)`, the orchestrator increments `rows_quarantined` but only creates a token and routes to a sink if `quarantine_destination` is set and exists in `PipelineConfig.sinks`.
- If `quarantine_destination` is missing or invalid, the row is skipped with no token/row created in the Landscape, producing an audit gap and silent data loss.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: any source that can quarantine rows
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into system 5 (engine) and look for bugs
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of quarantined-row path in orchestrator

## Steps To Reproduce

1. Implement/configure a source that yields `SourceRow.quarantined(row, destination=None)` or a destination sink name that is not present in `PipelineConfig.sinks`.
2. Run a pipeline.
3. Inspect the Landscape DB: the quarantined row is not present in `rows`/`tokens`, and no sink output exists for it.

## Expected Behavior

- Quarantined source rows should always be represented in the audit trail (at least as a `rows` record + a terminal token outcome), even if routing configuration is invalid.
- If a quarantine sink is configured, invalid destinations should be caught early (startup validation) and fail fast, not silently drop rows.

## Actual Behavior

- Rows can be silently dropped with no audit record if `quarantine_destination` is missing/unknown.

## Evidence

- Quarantined source path only creates a token when destination sink exists:
  - `src/elspeth/engine/orchestrator.py:589`
  - `src/elspeth/engine/orchestrator.py:594`
  - `src/elspeth/engine/orchestrator.py:606` (continues without creating row/token if no valid sink)

## Impact

- User-facing impact: quarantined rows disappear without trace; operators can’t inspect or remediate.
- Data integrity / security impact: violates audit completeness; quarantines are part of the legal/audit record.
- Performance or cost impact: debugging and reruns required to reconstruct what was dropped.

## Root Cause Hypothesis

- The quarantined row path was optimized to “route to sink if available” but does not enforce a fallback recording policy when destination is absent/invalid.

## Proposed Fix

- Code changes (modules/files):
  - Always create a row/token in Landscape for quarantined source rows.
  - Validate that `SourceRow.quarantine_destination` (if provided) exists in sinks; if not, either:
    - fail fast with a clear error, or
    - route to a default quarantine sink configured in settings, or
    - record as `RowOutcome.QUARANTINED` (discard) with a validation error record in Landscape.
- Config or schema changes:
  - Consider explicit `settings.datasource.quarantine_sink` default.
- Tests to add/update:
  - Add test: quarantined row with missing destination still appears in audit trail (and/or fails fast, per chosen policy).
- Risks or migration steps:
  - Ensure quarantine handling remains consistent with trust model (Tier 3 external data should not crash pipeline).

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` Tier 3 behavior (“quarantine rows that can’t be coerced/validated” and record why)
- Observed divergence: quarantined rows may not be recorded at all.
- Alignment plan or decision needed: define required minimum audit record for quarantined inputs.

## Acceptance Criteria

- Quarantined source rows are never silently dropped; they are either recorded and routed deterministically or the run fails fast with a clear configuration error.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_orchestrator.py`
  - `pytest tests/engine/test_integration.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
