# Bug Report: TokenManager.update_row_data drops lineage metadata

## Summary

- `update_row_data()` creates a new `TokenInfo` with only basic fields, dropping `fork_group_id`, `join_group_id`, and `expand_group_id` lineage metadata.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/engine/tokens.py:206-225` - `update_row_data()` creates new TokenInfo with only `row_id`, `token_id`, `row_data`, `branch_name`
- Missing: `fork_group_id`, `join_group_id`, `expand_group_id`
- `TokenInfo` contract at `src/elspeth/contracts/identity.py:10-32` defines these fields

## Impact

- User-facing impact: Lineage metadata lost after row data updates
- Data integrity: Fork/join/expand relationships not preserved

## Proposed Fix

- Preserve all TokenInfo fields when updating row_data

## Acceptance Criteria

- `update_row_data()` preserves fork_group_id, join_group_id, expand_group_id

## Resolution (2026-02-02)

**Status: FIXED**

### Root Cause

The problem was code duplication - `TokenInfo` construction was scattered across 8+ locations:
- `TokenManager.update_row_data()`
- `RowProcessor` (3 locations in processor.py)
- `TransformExecutor`, `GateExecutor`, `ConfigGateExecutor` (3 locations in executors.py)
- `AggregationExecutor` checkpoint serialization/deserialization (2 locations)

Each location manually constructed `TokenInfo` and independently decided which fields to copy, leading to inconsistent field preservation when `TokenInfo` gained new lineage fields.

### Fix Applied

1. **Added `TokenInfo.with_updated_data()` method** (`contracts/identity.py`)
   - Uses `dataclasses.replace()` to preserve ALL fields while updating only `row_data`
   - Future-proof: new fields automatically preserved

2. **Refactored all scattered constructions** to use the new method:
   - `TokenManager.update_row_data()` now delegates to `token.with_updated_data()`
   - `processor.py`: 3 locations refactored
   - `executors.py`: 3 locations refactored

3. **Fixed checkpoint serialization/deserialization** (`executors.py`)
   - `get_checkpoint_state()` now includes `fork_group_id`, `join_group_id`, `expand_group_id`
   - `restore_from_checkpoint()` now reads these fields

4. **Added comprehensive tests** (`tests/engine/test_tokens.py`)
   - `test_update_preserves_all_lineage_fields`
   - `test_update_preserves_expand_group_id`
   - `test_update_preserves_join_group_id`

### Files Modified

- `src/elspeth/contracts/identity.py` - Added `with_updated_data()` method
- `src/elspeth/engine/tokens.py` - Refactored `update_row_data()`
- `src/elspeth/engine/processor.py` - Refactored 3 TokenInfo constructions
- `src/elspeth/engine/executors.py` - Refactored 3 constructions + checkpoint format
- `tests/engine/test_tokens.py` - Added 3 test cases
- `tests/engine/test_aggregation_executor.py` - Adjusted checkpoint size test threshold
