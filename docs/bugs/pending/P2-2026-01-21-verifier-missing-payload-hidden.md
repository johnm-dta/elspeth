# Bug Report: CallVerifier hides missing payloads by substituting empty dicts

## Summary

- `CallVerifier` treats a missing response payload as `{}` by defaulting `get_call_response_data(...) or {}`. When payloads are purged or the payload store is disabled, verification proceeds against an empty baseline instead of reporting "missing recording" or "payload missing".

## Severity

- Severity: major
- Priority: P2

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
- Notable tool calls or steps: code inspection of verifier payload handling

## Steps To Reproduce

1. Run with payload store disabled or purge response payloads.
2. Invoke `CallVerifier.verify` for a call that was recorded as SUCCESS.
3. Observe verification proceeds with `recorded_response={}` and does not mark missing payloads.

## Expected Behavior

- Missing response payloads should be explicitly flagged as "payload missing" (similar to `ReplayPayloadMissingError`) and excluded from match/mismatch counts.

## Actual Behavior

- Missing payloads are silently treated as empty dicts and compared against live responses.

## Evidence

- Missing payloads defaulted to `{}`: `src/elspeth/plugins/clients/verifier.py:182-183`

## Impact

- User-facing impact: verification results are misleading; missing baselines can look like mismatches or false matches.
- Data integrity / security impact: audit grade should degrade to ATTRIBUTABLE_ONLY, but verifier does not surface this.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `CallVerifier` does not distinguish between "payload missing" and "payload is an empty dict".

## Proposed Fix

- Code changes (modules/files):
  - If `get_call_response_data` returns None for a SUCCESS call, mark a new flag (e.g., `payload_missing=True`) or surface a dedicated error.
  - Avoid counting these as matches/mismatches.
- Config or schema changes: none.
- Tests to add/update:
  - Add a verification test where payload store is disabled/purged and assert a missing-payload outcome.
- Risks or migration steps:
  - Update report metrics to include missing-payload counts.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (auditability standard: payload availability distinguishes replayable vs attributable-only)
- Observed divergence: verifier does not surface missing payloads.
- Reason (if known): convenience default.
- Alignment plan or decision needed: define how verify should behave with purged payloads.

## Acceptance Criteria

- Verification explicitly reports missing payloads and excludes them from normal diff comparisons.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k verifier_payload`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
