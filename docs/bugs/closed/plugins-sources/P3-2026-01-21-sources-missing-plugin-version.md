# Bug Report: CSVSource and JSONSource report plugin_version as 0.0.0

## RESOLUTION: 2026-02-02

**Status:** FIXED

**Fixed By:** Claude Code (Opus 4.5)

**Changes Made:**

1. **Source file changes:**
   - `src/elspeth/plugins/sources/csv_source.py`: Added `plugin_version = "1.0.0"` class attribute (line 60)
   - `src/elspeth/plugins/sources/json_source.py`: Added `plugin_version = "1.0.0"` class attribute (line 73)

2. **Tests added:**
   - `tests/plugins/sources/test_csv_source.py`: Added `test_has_plugin_version()` to TestCSVSource class
   - `tests/plugins/sources/test_json_source.py`: Added `test_has_plugin_version()` to TestJSONSource class
   - `tests/plugins/test_builtin_plugin_metadata.py`: Created new regression test file to verify all built-in plugins have non-default plugin_version

**Verification:**
- All 101 source plugin tests pass
- mypy type-checking passes
- ruff linting passes
- New tests verify CSVSource.plugin_version == "1.0.0" and JSONSource.plugin_version == "1.0.0"

**Note:** During fix implementation, discovered that `PassThrough` and `FieldMapper` transforms also lack explicit plugin_version. These are tracked separately as they are out of scope for this bug report.

---

## Summary

- `CSVSource` and `JSONSource` do not set `plugin_version`, so they inherit the base default `"0.0.0"`.
- The orchestrator records plugin metadata from the instance; audit records show incorrect versions for these core sources.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any run using CSVSource or JSONSource

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/sources`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Run a pipeline that uses `CSVSource` or `JSONSource`.
2. Inspect the `nodes.plugin_version` value in Landscape or export.

## Expected Behavior

- Source nodes record a real semantic version (e.g., `"1.0.0"`), consistent with other built-in plugins.

## Actual Behavior

- Source nodes record `"0.0.0"` because the class attribute is never set.

## Evidence

- `CSVSource` has no `plugin_version` attribute: `src/elspeth/plugins/sources/csv_source.py`
- `JSONSource` has no `plugin_version` attribute: `src/elspeth/plugins/sources/json_source.py`
- Base default is `"0.0.0"`: `src/elspeth/plugins/base.py:292-300`
- Orchestrator records instance metadata: `src/elspeth/engine/orchestrator.py:580-599`

## Impact

- User-facing impact: audit metadata for core sources is misleading or uninformative.
- Data integrity / security impact: weaker reproducibility guarantees (cannot tie outputs to source plugin version accurately).
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `CSVSource` and `JSONSource` omitted `plugin_version` while other built-ins set it explicitly.

## Proposed Fix

- Code changes (modules/files):
  - Add `plugin_version = "1.0.0"` (or actual version) to `CSVSource` and `JSONSource`.
- Config or schema changes: none.
- Tests to add/update:
  - Add a metadata test ensuring built-in sources expose non-default `plugin_version`.
- Risks or migration steps:
  - Update any tests that assume `0.0.0` (unlikely).

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (plugin_version required for auditability)
- Observed divergence: core sources report a placeholder version.
- Reason (if known): oversight during source implementation.
- Alignment plan or decision needed: define versioning policy for core plugins.

## Acceptance Criteria

- `CSVSource` and `JSONSource` expose explicit, non-default `plugin_version` values.
- Audit records show these versions for source nodes.

## Tests

- Suggested tests to run: `pytest tests/plugins/sources/test_csv_source.py`, `pytest tests/plugins/sources/test_json_source.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

---

## VERIFICATION: 2026-02-01

**Status:** STILL VALID

- `CSVSource` and `JSONSource` still lack explicit `plugin_version` attributes. (`src/elspeth/plugins/sources/csv_source.py:37-62`, `src/elspeth/plugins/sources/json_source.py:56-74`)
- Base classes still default to `plugin_version = "0.0.0"`. (`src/elspeth/plugins/base.py:65-68`)

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 6 (FINAL)

**Current Code Analysis:**

Examined the current implementation of both CSVSource and JSONSource:

- `src/elspeth/plugins/sources/csv_source.py` (lines 34-51): CSVSource class defines `name = "csv"` but has NO `plugin_version` attribute
- `src/elspeth/plugins/sources/json_source.py` (lines 34-51): JSONSource class defines `name = "json"` but has NO `plugin_version` attribute
- `src/elspeth/plugins/base.py` (line 489): BaseSource defines default `plugin_version: str = "0.0.0"`

Both sources inherit the default "0.0.0" version from BaseSource, confirming the bug report.

**Comparison with Other Plugins:**

Found that other built-in plugins DO set explicit plugin_version:
- `NullSource` (src/elspeth/plugins/sources/null_source.py:40): `plugin_version = "1.0.0"`
- `CSVSink` (src/elspeth/plugins/sinks/csv_sink.py:62): `plugin_version = "1.0.0"`
- `JSONSink` (src/elspeth/plugins/sinks/json_sink.py:53): `plugin_version = "1.0.0"`
- `DatabaseSink` (src/elspeth/plugins/sinks/database_sink.py:74): `plugin_version = "1.0.0"`
- `AzureBlobSink` (src/elspeth/plugins/azure/blob_sink.py:234): `plugin_version = "1.0.0"`
- `KeywordFilter` (src/elspeth/plugins/transforms/keyword_filter.py:70): `plugin_version = "1.0.0"`
- `AzureContentSafety` (src/elspeth/plugins/transforms/azure/content_safety.py:139): `plugin_version = "1.0.0"`
- `AzurePromptShield` (src/elspeth/plugins/transforms/azure/prompt_shield.py:111): `plugin_version = "1.0.0"`
- `AzureMultiQuery` (src/elspeth/plugins/llm/azure_multi_query.py:72): `plugin_version = "1.0.0"`

**Git History:**

Reviewed commit history:
- Commit `bba40c5` (2026-01-17): Added `plugin_version: str = "0.0.0"` default to BaseSource
- Commit `102ba4b`: CSVSource was created (predates plugin_version infrastructure)
- Commit `805bebd`: JSONSource was created (predates plugin_version infrastructure)
- Commit `1176136` (2026-01-20): NullSource was created WITH explicit `plugin_version = "1.0.0"` - showing the correct pattern
- Commit `7ee7c51` (most recent): Added self-validation to all builtin plugins but did NOT add plugin_version

No commits were found that added plugin_version to CSVSource or JSONSource.

**Root Cause Confirmed:**

YES - The bug is confirmed and still present. CSVSource and JSONSource:
1. Were created before the plugin_version infrastructure was fully established
2. Were never updated when NullSource and other plugins correctly set plugin_version
3. Currently inherit the placeholder "0.0.0" from BaseSource
4. Will record misleading audit metadata in the Landscape database

This creates an inconsistency where some core sources (NullSource) report real versions while others (CSVSource, JSONSource) report the placeholder default.

**Recommendation:**

**Keep open** - This is a valid audit integrity issue. The fix is straightforward:
1. Add `plugin_version = "1.0.0"` to both CSVSource (after line 50) and JSONSource (after line 50)
2. Add a test to verify all built-in plugins expose non-default plugin_version values
3. Consider adding a mypy or pytest validation rule to prevent future plugins from omitting plugin_version

The impact is minor (P3 priority is appropriate) but this should be fixed before 1.0 release to ensure audit trail consistency.
