# Bug Report: Azure multi-query guarantees non-existent base field

## Summary

- `get_llm_guaranteed_fields(spec.output_prefix)` returns base field that is never emitted. Multi-query emits `*_score`, `*_rationale` suffixed fields but also claims to guarantee the base field.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/plugins/llm/__init__.py:39` - `LLM_GUARANTEED_SUFFIXES` includes `""` (empty suffix)
- `src/elspeth/plugins/llm/azure_multi_query.py:158` - uses `get_llm_guaranteed_fields(spec.output_prefix)`
- Transform only emits suffixed fields (lines 450-477, 482-497)

## Impact

- User-facing impact: DAG contract validation may fail for valid pipelines
- Data integrity: Guaranteed fields claim is inaccurate

## Proposed Fix

- Multi-query should compute guaranteed fields from actual output_mapping suffixes

## Acceptance Criteria

- Guaranteed fields match actually emitted fields

## Verification (2026-02-01)

**Status: STILL VALID**

- `get_llm_guaranteed_fields()` still includes the base field (empty suffix) while Azure multi-query only emits suffixed fields. (`src/elspeth/plugins/llm/__init__.py:37-73`, `src/elspeth/plugins/llm/azure_multi_query.py:456-505`)
