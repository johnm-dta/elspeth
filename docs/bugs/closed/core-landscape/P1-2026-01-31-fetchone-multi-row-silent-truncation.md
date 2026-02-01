# Bug Report: execute_fetchone silently truncates multi-row results

## Summary

- `execute_fetchone()` uses `result.fetchone()` without checking for multiple rows, silently returning only the first row when multiple exist. This violates Tier 1 crash-on-anomaly requirements.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/core/landscape/_database_ops.py:25-29` uses `result.fetchone()` without checking for multiple rows
- Tier 1 audit data rules require crashing on anomalies, not silently truncating

## Impact

- User-facing impact: Queries expected to return single rows silently drop extra rows
- Data integrity / security impact: Audit queries could return incomplete/wrong data
- Performance or cost impact: None

## Root Cause Hypothesis

- Should use `result.one_or_none()` which raises `MultipleResultsFound` if >1 row exists.

## Proposed Fix

- Code changes:
  - Replace `result.fetchone()` with `result.one_or_none()` in `execute_fetchone()`
  - Or add explicit check: `rows = result.fetchall(); if len(rows) > 1: raise`
- Tests to add/update:
  - Add test that inserts duplicate data and asserts `execute_fetchone` raises

## Acceptance Criteria

- `execute_fetchone()` raises an error when query returns multiple rows
- Single-row and zero-row cases continue to work as before
