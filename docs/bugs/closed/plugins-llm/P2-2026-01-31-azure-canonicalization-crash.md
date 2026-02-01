# Bug Report: Unhandled canonicalization errors during template rendering crash Azure LLM transform

## Summary

- Only `TemplateError` is caught, but `render_with_metadata()` can raise `ValueError` from `canonical_json()` when row contains NaN/Infinity. This crashes the run instead of routing to on_error.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/plugins/llm/azure.py:282-293` - only catches TemplateError
- `src/elspeth/plugins/llm/templates.py:175` - `canonical_json(row)` can raise ValueError
- `src/elspeth/core/canonical.py:60-61` - NaN/Infinity raise ValueError

## Impact

- User-facing impact: Single row with NaN crashes entire run
- Data integrity: Violates Tier 2 trust model (row-scoped failure becomes run-scoped)

## Proposed Fix

- Add `except (ValueError, TypeError)` around `render_with_metadata()` and return `TransformResult.error()`

## Acceptance Criteria

- Rows with NaN/Infinity are routed to error, not crash the run

## Verification (2026-02-01)

**Status: STILL VALID**

- Azure LLM `_process_row` still only catches `TemplateError` around `render_with_metadata()`, so `ValueError` from canonicalization will crash. (`src/elspeth/plugins/llm/azure.py:287-299`, `src/elspeth/plugins/llm/templates.py:173-176`, `src/elspeth/core/canonical.py:60-61`)
