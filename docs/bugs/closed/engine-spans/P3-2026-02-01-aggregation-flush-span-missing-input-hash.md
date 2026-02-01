# Bug Report: Aggregation Flush Spans Missing input.hash

## Summary

- Aggregation flush spans no longer emit the `input.hash` attribute after switching from `transform_span()` to `aggregation_span()`, preventing trace-to-audit correlation for batch flushes.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex (from review)
- Date: 2026-02-01
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/engine/executors.py:1117-1144` computes `input_hash` for the aggregation batch, but `aggregation_span()` is called without any way to pass it.
- `src/elspeth/engine/spans.py:230-260` defines `aggregation_span()` without an `input_hash` parameter or `input.hash` attribute emission.
- `src/elspeth/engine/spans.py:145-179` shows `transform_span()` *does* emit `input.hash` when provided, which is the behavior aggregation flushes previously relied on.

## Impact

- Trace spans for aggregation flushes cannot be correlated to the batch input hash recorded in node_state/audit records.
- Operators lose the `input.hash` signal used to join telemetry/trace data with audit records for batch workflows.

## Proposed Fix

- Extend `aggregation_span()` to accept `input_hash` and set `input.hash` on the span.
- Pass the computed `input_hash` from `execute_flush()` into `aggregation_span()`.
- Add or update a span metadata test to verify `input.hash` is present on aggregation flush spans.

## Acceptance Criteria

- Aggregation flush spans include `input.hash` matching the computed `stable_hash(batch_input)`.
- Tests assert `input.hash` is present for aggregation spans.

## Verification (2026-02-01)

**Status: STILL VALID**

- `execute_flush()` computes `input_hash` but `aggregation_span()` has no input hash parameter, so `input.hash` is not emitted. (`src/elspeth/engine/executors.py:1117-1144`, `src/elspeth/engine/spans.py:230-260`)

## Resolution (2026-02-02)

**Status: FIXED**

**Root Cause:** When aggregation flushes were migrated from `transform_span()` to `aggregation_span()` (P2-2026-01-21 fix), the `input_hash` parameter was not carried over to the new span type.

**Fix Applied:**
1. Added `input_hash: str | None = None` parameter to `aggregation_span()` in `spans.py:225`
2. Added `input.hash` attribute emission when `input_hash` is provided in `spans.py:238-239`
3. Updated `execute_flush()` call site in `executors.py:1145` to pass the computed `input_hash`

**Test Added:**
- `test_aggregation_span_includes_input_hash` in `tests/engine/test_spans.py` verifies `input.hash` is emitted

**Verification:**
- All 36 span tests pass
- All 28 executor tests pass
- mypy type check passes
