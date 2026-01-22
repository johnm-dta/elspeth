# Bug Report: Coalesce timeout/incomplete failures are never recorded in audit

## Summary

- `flush_pending()` returns failure outcomes for `require_all` and `quorum` but does not record any audit state or token outcomes.
- `check_timeouts()` only merges when quorum is met; if quorum is not met at timeout, no failure is recorded and the pending entry persists.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (fix/rc1-bug-burndown-session-2)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into coalesce_executor, identify bugs, create bug docs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of coalesce executor and orchestrator

## Steps To Reproduce

1. Configure a coalesce with `policy: require_all` and a timeout.
2. Send only one branch token, then end the source (or hit timeout for `check_timeouts`).
3. Inspect the audit trail for failure records for the pending coalesce.

## Expected Behavior

- Missing branches should be recorded explicitly; failures should be recorded in audit (and missing branches quarantined if policy dictates).

## Actual Behavior

- Failures are returned to the caller but never recorded; the audit trail has no record of missing branches or timeout failures.

## Evidence

- `flush_pending()` creates failure outcomes without recording them: `src/elspeth/engine/coalesce_executor.py:421`
- `check_timeouts()` ignores quorum-not-met timeouts (no failure recorded): `src/elspeth/engine/coalesce_executor.py:357`
- Orchestrator assumes failures are recorded by executor: `src/elspeth/engine/orchestrator.py:878`
- Design expects missing branches to be recorded/quarantined: `docs/design/subsystems/00-overview.md#L322`

## Impact

- User-facing impact: explain/replay cannot show why a coalesce failed or which branches were missing.
- Data integrity / security impact: audit trail is incomplete; missing branches are not recorded.
- Performance or cost impact: pending entries can linger and grow without resolution.

## Root Cause Hypothesis

- Failure paths return `CoalesceOutcome` only and never call `LandscapeRecorder` to persist failure details.

## Proposed Fix

- Code changes (modules/files):
  - Record explicit failure outcomes for all arrived tokens when coalesce fails (require_all/quorum).
  - Record missing branches in audit metadata and consider quarantining or failure outcomes for missing branches per policy.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests for require_all/quorum timeout failure recording.
- Risks or migration steps:
  - Decide how missing branches are represented in audit (failure vs quarantine) and document the behavior.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/subsystems/00-overview.md#L322`
- Observed divergence: missing branches are not recorded; failures are silent.
- Reason (if known): executor returns outcomes but does not persist failure records.
- Alignment plan or decision needed: define failure recording semantics and implement them.

## Acceptance Criteria

- Coalesce failures create audit records indicating missing branches and policy decision.
- No pending coalesce remains after a timeout without a recorded resolution.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k failure`
- New tests required: yes (timeout failure recording)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`
