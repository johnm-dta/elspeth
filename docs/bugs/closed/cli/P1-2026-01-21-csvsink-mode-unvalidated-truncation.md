# Bug Report: CSVSink accepts invalid mode values and silently truncates

## Summary

- `CSVSinkConfig.mode` is an unconstrained string, so typos (e.g., "apend") are accepted and treated as write mode, silently truncating existing files.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6f088f467276582fa8016f91b4d3bb26c7 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive into src/elspeth/plugins/sinks for bugs.
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): Codex CLI, workspace-write sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Manual code inspection only

## Steps To Reproduce

1. Create a CSV file with existing data at `output.csv`.
2. Configure `CSVSink` with `mode: "apend"` (typo).
3. Call `sink.write([...], ctx)`.
4. Observe the file is opened in write mode and truncated.

## Expected Behavior

- Invalid mode values should be rejected during configuration validation.

## Actual Behavior

- Any non-"append" value falls back to write behavior, risking silent data loss.

## Evidence

- `src/elspeth/plugins/sinks/csv_sink.py` checks `if self._mode == "append"` and otherwise uses write mode.
- `CSVSinkConfig.mode` is declared as `str` with no validation.

## Impact

- User-facing impact: A simple typo can wipe existing output files.
- Data integrity / security impact: Data loss in audit artifacts.
- Performance or cost impact: None.

## Root Cause Hypothesis

- Config model does not constrain `mode` to allowed values.

## Proposed Fix

- Code changes (modules/files):
  - Change `CSVSinkConfig.mode` to `Literal["write", "append"]` and validate.
  - Consider raising explicit error for unknown values in `_open_file`.
- Config or schema changes: None.
- Tests to add/update:
  - Add config validation test for invalid mode values.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: Invalid config values are accepted silently.
- Reason (if known): Missing validation.
- Alignment plan or decision needed: Enforce strict config validation.

## Acceptance Criteria

- Invalid `mode` values raise `PluginConfigError` during initialization.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_csv_sink.py -k mode`
- New tests required: Add invalid-mode validation test.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## Verification (2026-01-24)

**Verifier:** Claude Sonnet 4.5
**Verification Date:** 2026-01-24
**Branch Verified:** fix/rc1-bug-burndown-session-4 (HEAD: 36e17f2)

### Status: **STILL VALID**

The bug remains unfixed in the current codebase. Detailed findings:

### Current Implementation Analysis

**File:** `/home/john/elspeth-rapid/src/elspeth/plugins/sinks/csv_sink.py`

**Line 31:** Mode field declaration
```python
mode: str = "write"  # "write" (truncate) or "append"
```

**Lines 168-199:** Mode validation logic in `_open_file()`
```python
if self._mode == "append" and self._path.exists():
    # ... append logic ...
else:
    # Write mode OR append to non-existent/empty file
    # Falls back to write mode for ANY non-"append" value
```

**Vulnerability confirmed:**
1. `CSVSinkConfig.mode` is typed as unconstrained `str`
2. No Pydantic validator constrains allowed values
3. No Literal type annotation (`Literal["write", "append"]`)
4. The `_open_file()` method uses equality check (`== "append"`) with implicit else-fallback to write mode
5. Typos like `"apend"`, `"appen"`, `"writes"` would silently truncate files

### Git History Analysis

**Append mode introduced:** commit `3eaebf6` (2026-01-20)
- Added `mode: str = "write"` field with comment indicating two valid values
- No validation was added at that time

**Changes since bug report (2026-01-21):**
- Only documentation and refactoring changes (diff checked via `git diff 3eaebf6..HEAD`)
- No validation logic added
- No Literal type annotation added
- No tests for invalid mode values

**Changes since 2026-01-21:**
```bash
$ git log --all --oneline --since="2026-01-21" -- src/elspeth/plugins/sinks/csv_sink.py
c786410 ELSPETH - Release Candidate 1
```

Only one commit touching the file, which was the RC-1 release merge. No validation fixes.

### Test Coverage Gap

Checked test files:
- `/home/john/elspeth-rapid/tests/plugins/sinks/test_csv_sink.py`
- `/home/john/elspeth-rapid/tests/plugins/sinks/test_csv_sink_append.py`

**No tests found for:**
- Invalid mode value rejection (e.g., `mode: "apend"`)
- Pydantic validation of mode field
- Explicit error on unknown mode values

### Reproduction Scenario

```python
# This configuration would be accepted:
config = {
    "path": "output.csv",
    "schema": {"fields": "dynamic"},
    "mode": "apend"  # TYPO - should be "append"
}

# CSVSinkConfig.from_dict(config) succeeds
# File opens in write mode, silently truncating existing data
```

### Impact Assessment

**Severity remains:** Major (P1)

**Risk scenarios:**
1. User typo in config file (`mode: "apend"` instead of `"append"`)
2. Resume operation fails, existing audit artifacts overwritten
3. Silent data loss - no error raised, pipeline continues normally
4. Audit trail integrity violation - loss of historical sink outputs

**ELSPETH auditability standard violation:**
> "I don't know what happened" is never an acceptable answer for any output

If a file is silently truncated due to typo, the audit trail would show the pipeline "succeeded" but data would be lost with no error record.

### Proposed Fix Validation

The proposed fix from the original bug report remains correct:

1. **Change field type to Literal:**
   ```python
   from typing import Literal

   class CSVSinkConfig(PathConfig):
       mode: Literal["write", "append"] = "write"
   ```

2. **Pydantic will automatically reject invalid values at config parse time**
   - `mode: "apend"` → `ValidationError: Input should be 'write' or 'append'`
   - No runtime check needed in `_open_file()` (though defensive check is fine)

3. **Add test for validation:**
   ```python
   def test_invalid_mode_rejected():
       with pytest.raises(ValidationError, match="write|append"):
           CSVSinkConfig.from_dict({
               "path": "output.csv",
               "schema": {"fields": "dynamic"},
               "mode": "apend"  # Invalid
           })
   ```

### Recommendation

**Action required:** Fix should be implemented before RC-1 final release.

**Priority justification:**
- Data loss risk in production
- Violates core auditability principle
- Simple fix (single line + test)
- No migration required (existing configs with valid values unaffected)

**No blocking dependencies:** This fix is independent and can be implemented immediately.
