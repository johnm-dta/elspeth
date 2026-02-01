# Bug Report: Quarantined rows silently dropped when quarantine destination is invalid

## Summary

- When `quarantine_sink` is invalid (typo, missing config), the code silently skips all handling - no token created, no outcome recorded, row is lost.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/engine/orchestrator.py:1198-1200` - `if quarantine_sink and quarantine_sink in config.sinks:` is a conditional guard
- If condition fails, code skips silently
- Violates "no silent drops" (CLAUDE.md:637-647)

## Impact

- User-facing impact: Rows disappear without trace on config error
- Data integrity: Silent data loss

## Proposed Fix

- Crash per "system-owned plugin bugs must crash" if quarantine_sink is configured but invalid

## Acceptance Criteria

- Invalid quarantine_sink causes immediate failure, not silent skip

## Verification (2026-02-01)

**Status: STILL VALID**

- Quarantine handling still short-circuits when `quarantine_sink` is unset or invalid with no failure/recording path. (`src/elspeth/engine/orchestrator.py:1198-1233`)

## Resolution (2026-02-02)

**Status: FIXED**

### Fix Applied

Added runtime validation in `orchestrator.py:1197-1213` that raises `RouteValidationError` when:
1. `quarantine_destination` is `None` or empty (missing destination)
2. `quarantine_destination` not in `config.sinks` (invalid destination)

The fix replaces the silent skip pattern with explicit crash-on-plugin-bug behavior, aligning with CLAUDE.md's "plugin bugs must crash" and "no silent drops" principles.

### Tests Added

- `tests/engine/test_orchestrator_errors.py::TestQuarantineDestinationRuntimeValidation::test_invalid_quarantine_destination_at_runtime_crashes`
- `tests/engine/test_orchestrator_errors.py::TestQuarantineDestinationRuntimeValidation::test_none_quarantine_destination_at_runtime_crashes`

### Verification

```bash
.venv/bin/python -m pytest tests/engine/test_orchestrator_errors.py::TestQuarantineDestinationRuntimeValidation -v
# 2 passed
```
