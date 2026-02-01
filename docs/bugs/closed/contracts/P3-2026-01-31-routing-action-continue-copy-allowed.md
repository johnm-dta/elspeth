# Bug Report: RoutingAction allows CONTINUE with COPY mode

## Summary

- `RoutingAction.__post_init__` doesn't validate that CONTINUE must use MOVE mode. Direct construction with COPY mode succeeds despite docstring saying "COPY is ONLY valid for FORK_TO_PATHS".

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/contracts/routing.py:63-81` - no check that CONTINUE requires MOVE
- Docstring at line 42-43 states "COPY is ONLY valid for FORK_TO_PATHS"
- Factory method `continue_()` at lines 83-90 defaults to MOVE (correct)

## Proposed Fix

- Add validation: `if kind == CONTINUE and mode == COPY: raise ValueError`

## Acceptance Criteria

- CONTINUE with COPY mode raises ValueError

## Verification (2026-02-01)

**Status: STILL VALID**

- `RoutingAction.__post_init__()` still lacks a guard for `CONTINUE` + `COPY` mode. (`src/elspeth/contracts/routing.py:63-80`)
