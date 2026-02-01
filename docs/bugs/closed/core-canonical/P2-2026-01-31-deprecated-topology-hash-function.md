# Bug Report: Deprecated compute_upstream_topology_hash violates no-legacy policy

## Summary

- `compute_upstream_topology_hash` function exists and is documented as "kept for backwards compatibility" which violates CLAUDE.md's No Legacy Code Policy.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/canonical.py:214-224` - docstring explicitly says the function is kept for backwards compatibility.
- CLAUDE.md:797-834 explicitly forbids backwards compatibility code

## Impact

- User-facing impact: None directly
- Policy: Violates No Legacy Code Policy

## Proposed Fix

- Remove function and update all call sites to use the current API

## Acceptance Criteria

- No backwards compatibility functions remain
- All call sites updated

## Verification (2026-02-01)

**Status: STILL VALID**

- `compute_upstream_topology_hash()` remains and still documents backwards compatibility usage. (`src/elspeth/core/canonical.py:214-224`)

## Resolution (2026-02-02)

**Status: CLOSED**

- Deleted `compute_upstream_topology_hash()` function from `src/elspeth/core/canonical.py`
- Updated `tests/core/checkpoint/test_manager.py` to use `compute_full_topology_hash(mock_graph)` instead
- No production code used the function (only the test file had a call site)
- Test was also corrected: it was verifying against the deprecated function while `CheckpointManager` actually uses `compute_full_topology_hash()`
