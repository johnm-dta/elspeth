# Bug Report: Schema compatibility allows int->float coercion even when consumer schema is strict

## Summary

- `_types_compatible()` unconditionally allows `int -> float` coercion without checking consumer schema strictness, violating the Data Manifesto rule that transforms/sinks must NOT coerce.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/contracts/data.py:225-245` - `_types_compatible()` unconditionally allows `int -> float` (lines 243-245).
- `src/elspeth/contracts/data.py:134-187` - `check_compatibility()` never inspects consumer schema strictness before calling `_types_compatible()`.
- CLAUDE.md Data Manifesto: transforms/sinks must NOT coerce

## Impact

- User-facing impact: Invalid pipelines pass DAG validation, fail at runtime
- Data integrity: Schema contract violations not caught at construction time

## Proposed Fix

- Add consumer strictness check to `_types_compatible()` before allowing int->float

## Acceptance Criteria

- Strict schemas reject int->float coercion
- Non-strict schemas continue to allow it

## Verification (2026-02-01)

**Status: STILL VALID**

- `_types_compatible()` still allows `int -> float` with no strictness gating, and `check_compatibility()` does not consider consumer schema strictness. (`src/elspeth/contracts/data.py:134-187`, `src/elspeth/contracts/data.py:225-245`)
