# Bug Report: --json-logs produces mixed JSON and plain-text lines

## Summary

- When `json_output=True`, stdlib logging modules bypass structlog and emit plain text, resulting in mixed JSON/plain-text log streams.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/logging.py:26-31` - stdlib logging configured with `format="%(message)s"`.
- `src/elspeth/core/logging.py:57-62` - uses `structlog.PrintLoggerFactory()` which prints directly.
- Modules using `logging.getLogger(__name__)` emit plain text even when JSON enabled

## Impact

- User-facing impact: Log parsing fails in CI/observability pipelines
- Data integrity: None

## Proposed Fix

- Use `structlog.stdlib.LoggerFactory()` and `ProcessorFormatter` to route stdlib loggers through structlog

## Acceptance Criteria

- All log output is valid JSON when `--json-logs` enabled

## Resolution (2026-02-01)

**Status: FIXED**

### Root Cause

`structlog.PrintLoggerFactory()` creates loggers that bypass stdlib entirely - they print directly to stdout. Meanwhile, `logging.basicConfig()` sets up stdlib loggers that also print to stdout with `format="%(message)s"`. There was no integration between them.

When code uses `logging.getLogger(__name__)`, those messages go through stdlib's handler (plain text) instead of structlog's processor chain (JSON).

### Fix Applied

Replaced `PrintLoggerFactory()` with `structlog.stdlib.LoggerFactory()` and configured stdlib logging to use `ProcessorFormatter`. This routes ALL log records (both structlog and stdlib) through the same processor chain.

Key changes to `src/elspeth/core/logging.py`:
1. Import `ProcessorFormatter` from `structlog.stdlib`
2. Configure structlog with `LoggerFactory()` and `wrap_for_formatter`
3. Configure stdlib with a handler using `ProcessorFormatter`
4. Added `_remove_internal_fields` processor to clean `_record` and `_from_structlog` from output

### Tests Added

- `test_stdlib_loggers_emit_json_when_json_output_enabled` - verifies stdlib loggers produce JSON
- `test_stdlib_loggers_emit_console_when_console_mode` - verifies console mode still works
