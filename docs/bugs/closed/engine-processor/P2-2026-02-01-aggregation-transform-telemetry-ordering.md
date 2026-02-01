# Bug Report: TransformCompleted emitted after TokenCompleted for transform-mode aggregation flushes

## Summary

- In transform-mode aggregations, buffered tokens emit `TokenCompleted` when they are first buffered, but the new per-token `TransformCompleted` telemetry fires only when the batch flushes. This reverses the expected order (TransformCompleted before TokenCompleted) for any token buffered earlier in the batch.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-02-01
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked
- Python version: not checked
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: review telemetry ordering regression after aggregation flush telemetry fix
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection only

## Steps To Reproduce

1. Configure a batch-aware aggregation with `output_mode=transform` and a count/timeout trigger.
2. Run with a batch size > 1 (or a timeout flush).
3. Observe telemetry for a token that was buffered before the flush triggers.

## Expected Behavior

- For each token, `TransformCompleted` should be emitted before `TokenCompleted` (or not at all if the token is already terminal and TransformCompleted is skipped).
- Telemetry ordering invariants remain consistent across all aggregation modes.

## Actual Behavior

- Buffered tokens emit `TokenCompleted` when they are marked `CONSUMED_IN_BATCH` on the non-flush path.
- When the batch later flushes, per-token `TransformCompleted` is emitted for all buffered tokens, including those already terminal, resulting in `TransformCompleted` arriving *after* `TokenCompleted`.

## Evidence

- Non-flush path in transform mode records `CONSUMED_IN_BATCH` and emits `TokenCompleted`: `src/elspeth/engine/processor.py:1016-1045`.
- Flush success path emits `TransformCompleted` for all buffered tokens:
  - Count-triggered flush: `src/elspeth/engine/processor.py:833-842`.
  - Timeout/end-of-source flush: `src/elspeth/engine/processor.py:552-560`.

## Impact

- User-facing impact: telemetry consumers that assume TransformCompleted precedes TokenCompleted may misorder events or treat late TransformCompleted as anomalous.
- Data integrity / security impact: none for Landscape; telemetry ordering correctness is compromised.
- Performance or cost impact: none directly.

## Root Cause Hypothesis

- Transform-mode buffering emits terminal `TokenCompleted` immediately, while transform telemetry is deferred to flush. The new per-token TransformCompleted emission did not account for already-terminal tokens.

## Proposed Fix

- Option A: Defer `TokenCompleted` for transform-mode buffered tokens until flush so TransformCompleted can precede it.
- Option B: Skip `TransformCompleted` for tokens already terminal (only emit for the triggering token or for tokens that had not yet emitted TokenCompleted).
- Option C: Emit TransformCompleted at buffer time for transform-mode so ordering is preserved.

## Acceptance Criteria

- For any token in transform-mode aggregation, `TransformCompleted` is emitted before `TokenCompleted`.
- Telemetry ordering is consistent across count, timeout, and end-of-source flush paths.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor.py -k aggregation_telemetry`
- New tests required: yes, add a test that asserts TransformCompleted-before-TokenCompleted ordering for transform-mode aggregation batches > 1 (including timeout flush).

## Notes / Links

- Related change: `docs/bugs/closed/engine-processor/P3-2026-01-31-aggregation-flush-missing-telemetry.md`

## Resolution (2026-02-02)

**Status: FIXED**

**Approach:** Option A - Defer `TokenCompleted` for transform-mode buffered tokens until flush time.

**Code Changes (5 locations in `src/elspeth/engine/processor.py`):**

1. **Buffer path (lines 1078-1082):** Removed `TokenCompleted` emission at buffer time. Tokens are recorded as `CONSUMED_IN_BATCH` in Landscape immediately (audit trail), but telemetry is deferred to flush.

2. **Count-trigger success (lines 1017-1023):** Added `TokenCompleted` emission for all buffered tokens after `TransformCompleted` loop completes.

3. **Count-trigger error (lines 824-831):** Added `TokenCompleted` emission for buffered tokens on failed flush. Even when flush fails, tokens have terminal state that needs telemetry.

4. **Timeout/EOS success (lines 633-636):** Same pattern as count-trigger - emit `TokenCompleted` after `TransformCompleted`.

5. **Timeout/EOS error (lines 539-543):** Added `TokenCompleted` emission for error path.

**Documentation Added:**

- Temporal decoupling comment in `_process_batch_aggregation_node` docstring (lines 726-743) explaining that Landscape recording and telemetry emission intentionally happen at different times for transform-mode aggregation.

**Tests Added (4 new tests in `tests/engine/test_processor_telemetry.py`):**

1. `test_transform_mode_aggregation_ordering_bug` - Core fix validation
2. `test_transform_mode_aggregation_batch_size_one_ordering` - Edge case: batch_size=1
3. `test_passthrough_mode_no_ordering_issue` - Negative test confirming passthrough mode unaffected
4. `test_transform_mode_failed_flush_emits_token_completed` - Error path telemetry

**Test Results:**

- 786 engine tests pass
- 16 telemetry tests pass
- mypy: no issues
- ruff: all checks passed

**ARB Review:** Approved by 4 specialist agents (architecture, python, quality, systems thinking) with all feedback implemented.
