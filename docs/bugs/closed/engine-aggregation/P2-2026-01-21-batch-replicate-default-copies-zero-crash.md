# Bug Report: BatchReplicate default_copies can be 0, causing success_multi([]) crash

## Summary

- BatchReplicate allows default_copies <= 0; when configured, the transform can emit zero output rows and then raises ValueError because TransformResult.success_multi forbids empty output.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked (workspace sandbox)
- Python version: not checked
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/transforms for bugs
- Model/version: GPT-5 Codex
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: reviewed BatchReplicate process logic

## Steps To Reproduce

1. Configure batch_replicate with default_copies: 0 (or a negative value).
2. Run aggregation with batch_replicate on any non-empty input batch.
3. Observe TransformResult.success_multi raises ValueError because output_rows is empty.

## Expected Behavior

- Configuration should reject default_copies < 1, or the transform should return a proper error result instead of crashing.

## Actual Behavior

- The transform builds zero output rows and then raises ValueError from TransformResult.success_multi.

## Evidence

- default_copies is not validated for minimum value: src/elspeth/plugins/transforms/batch_replicate.py:25-36
- copies loop uses range(copies) and can produce zero outputs: src/elspeth/plugins/transforms/batch_replicate.py:122-137
- success_multi([]) is invalid per protocol: docs/contracts/plugin-protocol.md:365-369

## Impact

- User-facing impact: pipeline crashes on valid-looking configuration.
- Data integrity / security impact: none (crash), but prevents processing.
- Performance or cost impact: wasted runs and retries.

## Root Cause Hypothesis

- BatchReplicateConfig does not constrain default_copies, and process() assumes at least one output row per input.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/batch_replicate.py
- Config or schema changes: validate default_copies >= 1 (Field with ge=1) and/or raise TransformResult.error when copies < 1.
- Tests to add/update: add config validation test and process test for invalid default_copies.
- Risks or migration steps: existing configs with default_copies <= 0 will now fail fast (intended).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/contracts/plugin-protocol.md:365-369 (success_multi([]) invalid)
- Observed divergence: transform can generate empty output_rows but still calls success_multi.
- Reason (if known): default_copies lacks validation.
- Alignment plan or decision needed: enforce minimum copies or return a proper error.

## Acceptance Criteria

- default_copies < 1 is rejected at config validation or converted into a TransformResult.error.
- BatchReplicate never calls success_multi([]).

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/test_batch_replicate.py
- New tests required: yes, config validation and empty-output guard.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/contracts/plugin-protocol.md

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 6a

**Current Code Analysis:**

Examined `/home/john/elspeth-rapid/src/elspeth/plugins/transforms/batch_replicate.py` (current HEAD):

1. **Configuration validation (lines 25-38):** `default_copies` field has NO minimum constraint:
   ```python
   default_copies: int = Field(
       default=1,
       description="Default number of copies if copies_field is missing or invalid",
   )
   ```
   - Missing `ge=1` constraint that would reject default_copies <= 0 at config load time
   - Tested with Pydantic: values of 0 and -5 are accepted without error

2. **Runtime protection INCOMPLETE (lines 133-141):**
   ```python
   copies = self._default_copies  # Line 133: If default_copies=0, copies=0
   if self._copies_field in row:
       raw_copies = row[self._copies_field]
       try:
           copies = int(raw_copies)
           if copies < 1:  # Line 138
               copies = self._default_copies  # Line 139: Falls back to 0!
       except (TypeError, ValueError):
           copies = self._default_copies  # Also falls back to 0!
   ```
   - The check `if copies < 1: copies = self._default_copies` creates a circular fallback
   - When `self._default_copies = 0`, the fallback doesn't help
   - This protects against row data having copies<1 ONLY when default_copies is valid (>=1)

3. **Empty output_rows scenario (line 144-151):**
   ```python
   for copy_idx in range(copies):  # range(0) = zero iterations
       output = dict(row)
       if self._include_copy_index:
           output["copy_index"] = copy_idx
       output_rows.append(output)

   return TransformResult.success_multi(output_rows)  # ValueError if empty
   ```
   - If ALL rows in batch have missing/invalid `copies_field` AND `default_copies=0`, then `output_rows` remains empty
   - `TransformResult.success_multi([])` raises ValueError per line 116-117 of `/home/john/elspeth-rapid/src/elspeth/contracts/results.py`

**Git History:**

- Original implementation: commit `a43f20a` (feat(plugins): add batch_replicate transform for deaggregation)
- Only change since creation: commit `7ee7c51` (feat: add self-validation to all builtin plugins) - added `_validate_self_consistency()` method but did NOT add validation for default_copies
- The runtime check at lines 138-139 was present from the original commit
- NO commits have addressed the default_copies validation issue

**Root Cause Confirmed:**

YES, bug is still present. The root cause remains:

1. **Primary issue:** `BatchReplicateConfig.default_copies` has no Pydantic constraint (missing `ge=1`)
2. **Secondary issue:** Runtime fallback logic at line 138-139 is circular - falls back to the invalid default_copies value itself
3. **Crash trigger:** When `default_copies=0` and a batch has rows without valid `copies_field` values, `output_rows` becomes empty, causing `success_multi([])` to raise ValueError

**Reproduction scenario:**
```yaml
aggregations:
  - name: replicate_batch
    plugin: batch_replicate
    trigger:
      count: 5
    output_mode: transform
    options:
      schema:
        fields: dynamic
      copies_field: quantity
      default_copies: 0  # ← Config validation accepts this
```

With input batch where `quantity` field is missing → crash with "success_multi requires at least one row"

**Recommendation:**

Keep open - bug confirmed valid and unfixed. Two-part fix needed:

1. **Config validation:** Add `ge=1` constraint to `default_copies` field in `BatchReplicateConfig`
2. **Runtime safety:** Either:
   - Fix circular fallback at lines 138-139 to use hardcoded minimum (e.g., `max(copies, 1)`)
   - OR guard before `success_multi()` call to handle edge case if config validation somehow fails

Priority should remain P2 - this is a configuration error that would be caught quickly in testing, but represents a crash-on-invalid-config scenario that violates principle of graceful degradation.

---

## CLOSURE: 2026-01-28

**Status:** FIXED

**Fixed By:** Claude Code

**Resolution:**

Added `ge=1` constraint to `default_copies` field in `BatchReplicateConfig` (line 38):

```python
default_copies: int = Field(
    default=1,
    ge=1,  # Added constraint
    description="Default number of copies if copies_field is missing or invalid",
)
```

**Note on runtime safety:** The runtime check at lines 137-141 (`if raw_copies < 1: raise ValueError`) already prevents invalid copies values from row data. The fix ensures config validation catches `default_copies < 1` at load time, preventing the circular fallback issue.

**Files Changed:**

- `src/elspeth/plugins/transforms/batch_replicate.py` - Added `ge=1` constraint
- `tests/plugins/transforms/test_batch_replicate.py` - Added config validation tests

**Tests Added:**

- `TestBatchReplicateConfigValidation::test_default_copies_zero_rejected`
- `TestBatchReplicateConfigValidation::test_default_copies_negative_rejected`

**Verification:**

```bash
pytest tests/plugins/transforms/test_batch_replicate.py -v -k config_validation
# Both tests pass - PluginConfigError raised for default_copies=0 and default_copies=-1
```
