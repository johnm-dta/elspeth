# Bug Report: Unknown custom_id in Azure batch output can crash during call recording

## Summary

- Azure batch result processing validates JSON structure but doesn't validate `custom_id` against checkpoint's `row_mapping`. Unknown custom_ids cause KeyError when accessing `requests_data`.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/plugins/llm/azure_batch.py:732-772` - validates JSON but not membership in row_mapping/requests
- Line 881 `original_request = requests_data[custom_id]` - KeyError if unknown
- Requires corrupted/unexpected Azure response (rare)

## Impact

- User-facing impact: Crash on malformed Azure response
- Data integrity: None (fails safe)

## Proposed Fix

- Add membership check before storing in `results_by_id`

## Acceptance Criteria

- Unknown custom_ids handled gracefully with clear error

## Verification (2026-02-01)

**Status: STILL VALID**

- Batch output parsing still accepts any `custom_id` and later indexes `requests_data[custom_id]` without membership check. (`src/elspeth/plugins/llm/azure_batch.py:732-772`, `src/elspeth/plugins/llm/azure_batch.py:881-883`)
