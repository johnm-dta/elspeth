# Bug Report: CSVSink append mode ignores explicit schema when headers differ

## Summary

- In append mode, CSVSink reads existing CSV headers and uses them without validating against the configured explicit schema, allowing schema drift or late failures when new fields appear.

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

1. Create `output.csv` with header `id` (no `score` column).
2. Configure CSVSink with schema `{mode: "free", fields: ["id: int", "score: float?"]}` and `mode: "append"`.
3. Write a row that includes `score`.
4. Observe a runtime `ValueError` (extra field) or silent omission if data is coerced elsewhere.

## Expected Behavior

- Append mode should validate that existing headers match the configured explicit schema and fail early with a clear error if they do not.

## Actual Behavior

- Existing headers are accepted as authoritative, ignoring the explicit schema.

## Evidence

- `src/elspeth/plugins/sinks/csv_sink.py` reads `existing_fieldnames` and uses them directly in append mode without comparing to `schema_config.fields`.

## Impact

- User-facing impact: Append runs can fail late or silently drop schema-defined fields.
- Data integrity / security impact: Output can drift from the declared schema, undermining auditability.
- Performance or cost impact: Wasted run time before failure.

## Root Cause Hypothesis

- Append path prioritizes file headers over explicit schema, with no validation step.

## Proposed Fix

- Code changes (modules/files):
  - If schema is explicit, compare `existing_fieldnames` to schema field names and raise if mismatch.
  - Optionally allow a strict flag to force header rewrite (if allowed by policy).
- Config or schema changes: None.
- Tests to add/update:
  - Add append-mode test that asserts schema/header mismatch raises a clear error.
- Risks or migration steps: Existing append workflows with mismatched headers will start failing fast (desired).

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md` (schema is a contract for sinks).
- Observed divergence: Explicit schema is ignored when appending to existing file.
- Reason (if known): Append mode implemented without schema validation.
- Alignment plan or decision needed: Enforce schema compliance on append.

## Acceptance Criteria

- Append mode fails fast with a clear error when file headers do not match explicit schema.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_csv_sink_append.py -v`
- New tests required: Add schema mismatch coverage.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## Verification (2026-01-24)

**Status: STILL VALID**

### Code Analysis

Verified in `/home/john/elspeth-rapid/src/elspeth/plugins/sinks/csv_sink.py`:

**Append mode path (lines 168-186):**
```python
if self._mode == "append" and self._path.exists():
    # Try to read existing headers from file
    with open(self._path, encoding=self._encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=self._delimiter)
        existing_fieldnames = reader.fieldnames

    if existing_fieldnames:
        # Use existing headers, append mode (no header write)
        self._fieldnames = list(existing_fieldnames)
        self._file = open(...)
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._fieldnames,
            delimiter=self._delimiter,
        )
        # No header write - already exists
        return
```

**Problem confirmed:** When `mode == "append"` and file exists:
1. Code reads `existing_fieldnames` from file (line 172)
2. Uses them directly as `self._fieldnames` (line 176)
3. Returns immediately (line 186)
4. **NEVER calls** `_get_fieldnames_from_schema_or_row()` which would enforce schema

**Contrast with write mode (lines 189-199):**
- Calls `_get_fieldnames_from_schema_or_row(rows[0])` (line 190)
- This method checks `if not self._schema_config.is_dynamic` and uses schema field names (lines 209-211)

### Test Coverage Analysis

Examined `/home/john/elspeth-rapid/tests/plugins/sinks/test_csv_sink_append.py`:
- All tests use **dynamic schema** (`{"fields": "dynamic"}`)
- **NO tests** verify append mode with explicit schema (`mode: "strict"` or `mode: "free"`)
- **NO tests** verify header mismatch detection

Test file in `/home/john/elspeth-rapid/tests/plugins/sinks/test_csv_sink.py` has one explicit schema test (line 254), but it uses **write mode**, not append.

### Git History Analysis

```bash
git log --all --oneline --since="2026-01-21" -- src/elspeth/plugins/sinks/csv_sink.py
# Output: c786410 ELSPETH - Release Candidate 1
```

- File created on 2026-01-22 (RC1)
- **Zero commits** since RC1
- Bug reported 2026-01-21 (before RC1, based on inspection of ae2c0e6 commit)
- **No fixes applied**

### Reproduction Scenario

Following the steps from the bug report:

1. **Create file with subset of schema fields:**
   ```python
   # output.csv contains: id
   # 1
   ```

2. **Configure CSVSink with explicit schema:**
   ```python
   config = {
       "path": "output.csv",
       "schema": {"mode": "free", "fields": ["id: int", "score: float?"]},
       "mode": "append"
   }
   ```

3. **Write row with all schema fields:**
   ```python
   sink.write([{"id": 2, "score": 1.5}], ctx)
   ```

4. **Expected:** ValueError from csv.DictWriter (extra key "score")
5. **Actual:** No validation - accepts file headers, then fails at write time

### Impact Assessment

**Severity: Confirmed P1**

- **Audit integrity violation:** Output file can have different schema than declared in config
- **Late failure:** Fails during `DictWriter.writerow()` instead of at initialization
- **Silent data loss:** If using `extrasaction='ignore'`, would silently drop schema-defined fields
- **Schema drift:** No enforcement means append mode cannot guarantee schema compliance

### Root Cause Confirmation

Append mode path has **NO validation** that `existing_fieldnames` matches `schema_config.fields`.

The fix requires:
1. In `_open_file()`, after reading `existing_fieldnames` (line 172)
2. When schema is explicit (not dynamic), compare `existing_fieldnames` to schema field names
3. Raise clear error if mismatch detected

### Recommendation

**Fix required before production use.** Append mode with explicit schema is currently unsafe for audit-critical workflows.
