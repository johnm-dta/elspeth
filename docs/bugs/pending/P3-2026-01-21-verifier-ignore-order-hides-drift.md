# Bug Report: CallVerifier ignores list ordering by default, masking drift

## Summary

- `CallVerifier` hard-codes `ignore_order=True` in DeepDiff comparisons. For responses where list order matters (ranked results, tool calls, top-k outputs), order changes are ignored and drift can be missed.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/clients` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of verifier comparison defaults

## Steps To Reproduce

1. Record a response containing an ordered list (e.g., ranked results `["a", "b", "c"]`).
2. Verify against a live response with the list reordered (e.g., `["c", "b", "a"]`).
3. Observe verification reports a match because order is ignored.

## Expected Behavior

- Order-sensitive responses should detect reordering as drift by default or allow configuration to enforce ordering.

## Actual Behavior

- Ordering differences are ignored globally for all responses.

## Evidence

- Hard-coded `ignore_order=True`: `src/elspeth/plugins/clients/verifier.py:186-190`

## Impact

- User-facing impact: verification can miss real drift in ordered responses.
- Data integrity / security impact: baseline comparisons are weaker than intended.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- DeepDiff is configured with a global ignore-order setting rather than a per-response or configurable policy.

## Proposed Fix

- Code changes (modules/files):
  - Make `ignore_order` configurable and default to `False`, or allow per-call overrides.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test case for ordered list drift detection.
- Risks or migration steps:
  - Some existing comparisons may become more sensitive; document the change.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: verification ignores order changes for all responses.
- Reason (if known): simplified comparison.
- Alignment plan or decision needed: define verification semantics for ordered data.

## Acceptance Criteria

- Verification reports drift when ordered lists change unless explicitly configured to ignore ordering.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k verifier_order`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
