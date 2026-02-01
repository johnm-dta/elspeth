# Bug Report: NodeStateRepository does not enforce OPEN/PENDING invariant forbidden fields

## Summary

- `NodeStateRepository` returns OPEN and PENDING states without validating that `output_hash`, `completed_at`, and `duration_ms` are None as required by the state machine invariants.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/core/landscape/repositories.py:297-308` - OPEN states returned without validating forbidden fields are None
- Lines 310-330 - PENDING validation only checks required fields, not forbidden fields
- State machine requires OPEN/PENDING states to have no completion data

## Impact

- User-facing impact: Corrupted node_states would load without error
- Data integrity / security impact: Audit data corruption goes undetected
- Performance or cost impact: None

## Root Cause Hypothesis

- Missing explicit validation that `output_hash`, `completed_at`, `duration_ms` are None for OPEN/PENDING states.

## Proposed Fix

- Code changes:
  - Add validation in repository methods that OPEN/PENDING states have None for forbidden fields
  - Raise ValueError if invariants are violated
- Tests to add/update:
  - Add test with corrupted OPEN state (has output_hash), assert ValueError

## Acceptance Criteria

- OPEN states with non-None completion fields cause immediate failure
- PENDING states with non-None completion fields cause immediate failure
- Valid states continue to load correctly
