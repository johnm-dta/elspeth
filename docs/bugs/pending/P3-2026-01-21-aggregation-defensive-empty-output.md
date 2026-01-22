# Bug Report: Aggregation output defaults to empty row on contract violation

## Summary

- In aggregation `output_mode` single/transform, the processor substitutes `{}` when `result.row` is missing. This masks plugin contract violations and can emit empty rows instead of crashing, violating the repository’s “no defensive programming” rule.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/engine/processor.py for bugs; create reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Implement a batch-aware transform that incorrectly returns `TransformResult.success_multi(...)` while aggregation is configured as `output_mode: single`, or returns an invalid `TransformResult` with `row=None`.
2. Trigger a batch flush.

## Expected Behavior

- The processor should crash on the contract violation (single-mode expects a single row), not fabricate output.

## Actual Behavior

- The processor substitutes an empty dict and proceeds.

## Evidence

- Single mode fallback: `src/elspeth/engine/processor.py:224-227` uses `{}` if `result.row` is None.
- Transform mode fallback: `src/elspeth/engine/processor.py:323-325` uses `{}` if `result.row` is None.
- CLAUDE.md prohibits defensive patterns that mask plugin bugs.

## Impact

- User-facing impact: Silent emission of empty rows.
- Data integrity / security impact: Audit trail contains fabricated output.
- Performance or cost impact: Downstream steps process meaningless data.

## Root Cause Hypothesis

- Defensive defaults in aggregation output handling mask invalid TransformResult payloads.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`
- Config or schema changes: None
- Tests to add/update: Add tests that assert contract violations raise errors.
- Risks or migration steps: None; aligns with crash-on-bug policy.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md “no defensive programming”.
- Observed divergence: Processor substitutes `{}` instead of raising.
- Reason (if known): Convenience fallback added during aggregation implementation.
- Alignment plan or decision needed: Decide whether to enforce strict output-mode contracts.

## Acceptance Criteria

- Aggregation output-mode violations raise errors instead of emitting empty rows.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor.py -k aggregation_contract`
- New tests required: Yes (contract violation handling).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md
