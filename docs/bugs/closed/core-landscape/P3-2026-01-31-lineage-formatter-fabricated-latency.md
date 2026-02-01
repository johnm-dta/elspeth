# Bug Report: LineageTextFormatter fabricates 0.0ms latency when missing

## Summary

- `LineageTextFormatter` uses `latency = call.latency_ms if call.latency_ms is not None else 0.0`, fabricating data instead of showing "N/A".

## Severity

- Severity: trivial
- Priority: P3

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/landscape/formatters.py:173-174` - `else 0.0` fabricates data

## Proposed Fix

- Display "N/A" or omit latency when None

## Acceptance Criteria

- Missing latency shows as "N/A", not "0.0ms"

## Verification (2026-02-01)

**Status: STILL VALID**

- Formatter still substitutes `0.0` when `latency_ms` is missing. (`src/elspeth/core/landscape/formatters.py:173-174`)

## Closure Report (2026-02-01)

**Status:** CLOSED (IMPLEMENTED)

### Fix Summary

- Render missing call latency as "N/A" instead of fabricating `0.0ms`.

### Test Coverage

- `tests/core/landscape/test_formatters.py::TestLineageTextFormatter::test_formats_missing_latency_as_na`
