# Bug Report: Multi-query config allows output-key collisions that silently overwrite LLM results

## Summary

- Multi-query transform config allows duplicate case_study/criterion names and collisions with reserved suffixes, causing later values to silently overwrite earlier ones via `output.update()`.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31
- Related run/issue ID: N/A

## Evidence

- `src/elspeth/plugins/llm/multi_query.py:284-303` - `expand_queries()` builds `output_prefix = f"{case_study.name}_{criterion.name}"` without any uniqueness validation.
- `src/elspeth/plugins/llm/multi_query.py:234-242` - `output_mapping` validator only checks non-empty; no collision checks.
- `src/elspeth/plugins/llm/azure_multi_query.py:624-631` and `src/elspeth/plugins/llm/openrouter_multi_query.py:812-819` - results are merged via `output.update()`, which silently overwrites duplicate keys.
- Reserved suffixes exist for guaranteed/audit fields (`_usage`, `_model`, `_template_hash`, etc.) but are not protected from collision with user-defined suffixes. (`src/elspeth/plugins/llm/__init__.py:37-52`)

## Impact

- User-facing impact: Silent data loss - later LLM results overwrite earlier ones
- Data integrity / security impact: Audit trail contains incomplete/wrong data without any error
- Performance or cost impact: Wasted LLM calls for results that get overwritten

## Root Cause Hypothesis

- Missing validation for uniqueness of case_study.name + criterion.name combinations and collision detection with reserved suffixes.

## Proposed Fix

- Code changes:
  - Add `@model_validator(mode="after")` to MultiQuerySpec
  - Validate uniqueness of (case_study.name, criterion.name) pairs
  - Detect collisions with reserved suffixes (_usage, _model, _template_hash, _variables_hash)
- Tests to add/update:
  - Add config validation test with duplicate names, assert ValueError
  - Add test with suffix collision, assert ValueError

## Acceptance Criteria

- Duplicate case_study/criterion name combinations are rejected at config time
- Collisions with reserved suffixes are detected and rejected
- No silent overwrites possible in output

## Verification (2026-02-01)

**Status: FIXED**

- Added `@model_validator(mode="after")` to `MultiQueryConfig` that validates:
  1. No duplicate case_study names
  2. No duplicate criterion names
  3. No output_mapping suffixes that collide with reserved LLM suffixes (_usage, _model, _template_hash, etc.)
- Added 3 regression tests in `TestMultiQueryConfig`
- All 36 multi_query tests pass, mypy clean
