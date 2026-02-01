# Bug Report: Multiple transforms report plugin_version as 0.0.0

## RESOLUTION: 2026-02-02

**Status:** FIXED

**Fixed By:** Claude Code (Opus 4.5)

**Changes Made:**

1. **Transform file changes - Added `plugin_version = "1.0.0"` to:**
   - `src/elspeth/plugins/transforms/passthrough.py` (PassThrough class)
   - `src/elspeth/plugins/transforms/field_mapper.py` (FieldMapper class)
   - `src/elspeth/plugins/transforms/truncate.py` (Truncate class)
   - `src/elspeth/plugins/transforms/batch_stats.py` (BatchStats class)
   - `src/elspeth/plugins/transforms/batch_replicate.py` (BatchReplicate class)
   - `src/elspeth/plugins/transforms/json_explode.py` (JSONExplode class)

2. **Tests updated:**
   - `tests/plugins/test_builtin_plugin_metadata.py`: Added tests for all transforms

**Verification:**
- All 156 transform tests pass
- All 13 metadata tests pass
- mypy type-checking passes
- ruff linting passes

---

## Summary

- Multiple built-in transforms do not set `plugin_version`, so they inherit the base default `"0.0.0"`.
- The orchestrator records plugin metadata from the instance; audit records show incorrect versions for these transforms.
- Affected transforms: `PassThrough`, `FieldMapper`, `Truncate`, `BatchStats`, `BatchReplicate`, `JSONExplode`

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Claude Code (discovered during P3-2026-01-21 fix)
- Date: 2026-02-02
- Related run/issue ID: P3-2026-01-21-sources-missing-plugin-version (related bug for sources)

## Environment

- Commit/branch: `RC1-bugs-final`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any run using affected transforms

## Steps To Reproduce

1. Run a pipeline using any of the affected transforms (PassThrough, FieldMapper, Truncate, BatchStats, BatchReplicate, JSONExplode)
2. Inspect the `nodes.plugin_version` value in Landscape or export.

## Expected Behavior

- Transform nodes record a real semantic version (e.g., `"1.0.0"`), consistent with other built-in plugins.

## Actual Behavior

- Transform nodes record `"0.0.0"` because the class attribute is never set.

## Evidence

Transforms missing `plugin_version`:
- `PassThrough` - `src/elspeth/plugins/transforms/passthrough.py`
- `FieldMapper` - `src/elspeth/plugins/transforms/field_mapper.py`
- `Truncate` - `src/elspeth/plugins/transforms/truncate.py`
- `BatchStats` - `src/elspeth/plugins/transforms/batch_stats.py`
- `BatchReplicate` - `src/elspeth/plugins/transforms/batch_replicate.py`
- `JSONExplode` - `src/elspeth/plugins/transforms/json_explode.py`

Transforms WITH `plugin_version = "1.0.0"`:
- `KeywordFilter` - `src/elspeth/plugins/transforms/keyword_filter.py:70`
- `AzurePromptShield` - `src/elspeth/plugins/transforms/azure/prompt_shield.py:134`
- `AzureContentSafety` - `src/elspeth/plugins/transforms/azure/content_safety.py:162`

Base default is `"0.0.0"`: `src/elspeth/plugins/base.py:67`

## Impact

- User-facing impact: audit metadata for core transforms is misleading or uninformative.
- Data integrity / security impact: weaker reproducibility guarantees (cannot tie outputs to transform plugin version accurately).
- Performance or cost impact: N/A

## Root Cause Hypothesis

- These transforms were created before the `plugin_version` infrastructure was fully established.
- Same root cause as P3-2026-01-21-sources-missing-plugin-version.

## Proposed Fix

- Code changes (modules/files):
  - Add `plugin_version = "1.0.0"` to all affected transform classes.
- Config or schema changes: none.
- Tests to add/update:
  - Add metadata tests ensuring all built-in transforms expose non-default `plugin_version`.
- Risks or migration steps:
  - Update any tests that assume `0.0.0` (unlikely).

## Acceptance Criteria

- All affected transforms expose explicit, non-default `plugin_version` values.
- Audit records show these versions for transform nodes.

## Tests

- Suggested tests to run: `pytest tests/plugins/transforms/`
- New tests required: yes

## Notes / Links

- Related issues/PRs: P3-2026-01-21-sources-missing-plugin-version (same issue for sources - fixed)
- Related design docs: `docs/contracts/plugin-protocol.md`
