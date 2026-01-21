# Bug Report: Replay/verify collapses duplicate calls with identical request_hash

## Summary

- `CallReplayer` and `CallVerifier` match recordings only by `request_hash` and `call_type`. When the same request is made multiple times in a run, `find_call_by_request_hash` returns the first matching call, so replay/verify always uses the first response and ignores later calls.

## Severity

- Severity: major
- Priority: P1

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
- Notable tool calls or steps: code inspection of replay/verify lookup logic

## Steps To Reproduce

1. In a run, make the same external request twice with identical `request_data` (e.g., retries or multiple identical LLM calls).
2. Enter replay/verify mode for that run.
3. Observe replay/verify always returns/compares against the first recorded call.

## Expected Behavior

- Replay/verify should disambiguate repeated identical requests (e.g., by call order or call index) and return the matching response for each invocation.

## Actual Behavior

- The first recorded call is always used; later calls are ignored.

## Evidence

- Replay lookup uses only `request_hash`: `src/elspeth/plugins/clients/replayer.py:156-177`
- Recorder returns first match when duplicates exist: `src/elspeth/core/landscape/recorder.py:2503-2519`
- Verify lookup uses only `request_hash`: `src/elspeth/plugins/clients/verifier.py:159-166`

## Impact

- User-facing impact: replay returns incorrect responses; verify reports drift against the wrong baseline.
- Data integrity / security impact: reproducibility claims are undermined for repeated calls.
- Performance or cost impact: potential false positives/negatives in verification.

## Root Cause Hypothesis

- Lookup keys do not include call order or per-state sequence, so duplicates collapse to the earliest call.

## Proposed Fix

- Code changes (modules/files):
  - Introduce a per-request sequence cursor in `CallReplayer`/`CallVerifier`, or match by `(state_id, call_index)` when available.
  - Optionally return and consume calls in chronological order per request hash.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests where identical requests produce different responses and ensure replay/verify consumes them in order.
- Risks or migration steps:
  - Decide how to disambiguate repeated calls across different states or nodes.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: replay/verify cannot represent repeated identical calls.
- Reason (if known): lookup by request hash was simpler.
- Alignment plan or decision needed: define required replay semantics for duplicate requests.

## Acceptance Criteria

- Replay/verify distinguishes multiple identical requests and returns/compares each call in order.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k replay_duplicate`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
