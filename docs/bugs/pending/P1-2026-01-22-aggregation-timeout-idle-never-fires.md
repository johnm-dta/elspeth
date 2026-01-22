# Bug Report: Timeout triggers never fire during idle periods

## Summary

- Timeout-based aggregation triggers are only evaluated after new rows are accepted. If no new rows arrive, batches can exceed `timeout_seconds` indefinitely and never flush.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/triggers.py` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: traced trigger evaluation call sites in processor

## Steps To Reproduce

1. Configure an aggregation trigger with `timeout_seconds: 5` and no count trigger.
2. Buffer a few rows, then stop ingesting new rows while keeping the pipeline alive.
3. Wait > 5 seconds and observe that no flush happens until a new row arrives or the source ends.

## Expected Behavior

- The batch should flush once `timeout_seconds` elapses, even if no new rows arrive.

## Actual Behavior

- Timeout is only evaluated on `record_accept()`/`should_trigger()` calls, so idle batches never flush.

## Evidence

- Trigger evaluation only happens after buffering a row: `src/elspeth/engine/processor.py:183-187`
- Timeout condition relies on time elapsed: `src/elspeth/engine/triggers.py:100-103`
- Spec says timeout fires when duration elapses: `docs/contracts/plugin-protocol.md:1208-1210`

## Impact

- User-facing impact: outputs can be delayed indefinitely in low-traffic or bursty streams.
- Data integrity / security impact: audit trail lacks timely batch completion events.
- Performance or cost impact: buffers can grow unbounded, increasing memory use.

## Root Cause Hypothesis

- No scheduler or periodic timeout check exists; `should_trigger()` is invoked only during row processing.

## Proposed Fix

- Code changes (modules/files):
  - Add periodic timeout checks in orchestrator/processor loop.
  - Optionally expose a `next_deadline()` on `TriggerEvaluator` to schedule sleeps.
- Config or schema changes: none.
- Tests to add/update:
  - Integration test for timeout flush without new row arrivals.
- Risks or migration steps:
  - Ensure periodic checks do not trigger when buffer is empty.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md:1208-1213`
- Observed divergence: timeout is defined as elapsed duration but is only checked on new row accept.
- Reason (if known): trigger evaluation is tied to row processing.
- Alignment plan or decision needed: define whether timeouts must fire without new inputs.

## Acceptance Criteria

- Batches flush after `timeout_seconds` even when no new rows are accepted.

## Tests

- Suggested tests to run: `pytest tests/engine/test_triggers.py -k timeout`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
