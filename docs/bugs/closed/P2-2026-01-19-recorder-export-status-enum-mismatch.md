# Bug Report: LandscapeRecorder returns `Run.export_status` as a raw string (not `ExportStatus` enum) and can leave stale `export_error`

## Summary

- `LandscapeRecorder.get_run()` and `LandscapeRecorder.list_runs()` populate `Run.export_status` directly from the DB (`str | None`) instead of coercing to `ExportStatus | None`, violating the strict audit contract.
- `LandscapeRecorder.set_export_status()` accepts an unvalidated `status: str` and never clears `export_error` when transitioning to non-failed statuses, so a run can appear `export_status=completed` while still carrying an old error message.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `8ca061c9293db459c9a900f2f74b19b59a364a42`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive subsystem 4 (Landscape) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: minimal Python repros + code inspection

## Steps To Reproduce

1. Run the following Python snippet:
   - `python - <<'PY'`
   - `from elspeth.core.landscape.database import LandscapeDB`
   - `from elspeth.core.landscape.recorder import LandscapeRecorder`
   - `from elspeth.contracts.enums import ExportStatus`
   - `db = LandscapeDB.in_memory()`
   - `rec = LandscapeRecorder(db)`
   - `run = rec.begin_run(config={}, canonical_version="v1")`
   - `rec.set_export_status(run.run_id, "completed")`
   - `r = rec.get_run(run.run_id)`
   - `print(type(r.export_status), r.export_status, isinstance(r.export_status, ExportStatus))`
   - `PY`
2. Observe `export_status` is a `str`, not `ExportStatus`.
3. Optional stale error repro:
   1. `rec.set_export_status(run.run_id, "failed", error="boom")`
   2. `rec.set_export_status(run.run_id, "completed")`
   3. `rec.get_run(run.run_id).export_error` still contains `"boom"`.

## Expected Behavior

- `Run.export_status` is always `ExportStatus | None` when returned from Landscape APIs.
- `set_export_status()` validates status values against `ExportStatus` and maintains consistent fields:
  - `export_error` cleared when `export_status != failed`
  - `exported_at` only set when completed

## Actual Behavior

- `Run.export_status` is returned as a raw `str | None`.
- `export_error` can remain populated after status transitions to `completed`.

## Evidence

- Contract requires enum typing:
  - `src/elspeth/contracts/audit.py:27-46` (`Run.export_status: ExportStatus | None`)
- Recorder returns raw DB values:
  - `src/elspeth/core/landscape/recorder.py:327-341` (`export_status=row.export_status`)
  - `src/elspeth/core/landscape/recorder.py:366-383` (`export_status=row.export_status`)
- Status setter does not validate or clear stale fields:
  - `src/elspeth/core/landscape/recorder.py:385-415`

## Impact

- User-facing impact: TUI/explain/export tooling that expects `ExportStatus` (or uses `.value`) can break or mis-render export status.
- Data integrity / security impact: medium-high. Tier 1 audit data can contain invalid/unvalidated enum strings; later strict coercions will crash.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `LandscapeRecorder` bypasses the strict repository conversion pattern used elsewhere and returns raw DB values for `export_status`.
- `set_export_status()` implements a partial update without field consistency rules (no clearing on transition).

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/recorder.py`:
    - In `get_run()` / `list_runs()`, coerce `export_status` to `ExportStatus` when non-NULL.
    - Change `set_export_status()` to accept `ExportStatus | str`, coerce via `ExportStatus(...)`, and raise on invalid values.
    - Clear `export_error` when status transitions away from `failed` (unless explicitly provided).
- Config or schema changes: none.
- Tests to add/update:
  - Add tests asserting `isinstance(run.export_status, ExportStatus)` for:
    - `get_run()` after `set_export_status(...)`
    - `list_runs()` results after setting status
  - Add a test that verifies `export_error` is cleared when moving from `failed` to `completed`.
- Risks or migration steps:
  - Existing DBs with invalid `runs.export_status` strings will now crash on read (desired per Data Manifesto); provide an actionable error message if needed.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (“Tier 1: Our Data - crash on invalid enum/type”)
- Observed divergence: audit DB enum fields can be silently invalid and contracts are not respected at read boundaries.
- Reason (if known): `LandscapeRecorder` duplicates repository logic and missed enum coercion + update invariants.
- Alignment plan or decision needed: none.

## Acceptance Criteria

- `LandscapeRecorder.get_run()` and `list_runs()` always return `Run.export_status` as `ExportStatus | None`.
- `set_export_status()` rejects invalid status values and does not leave stale `export_error` on success.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_recorder.py`
- New tests required: yes (export_status type + stale export_error)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md` (audit trail integrity + export tracking)

## Resolution

**Status:** CLOSED (2026-01-21)
**Resolved by:** Claude Opus 4.5

### Changes Made

**Code fix (`src/elspeth/core/landscape/recorder.py`):**

1. **Added `ExportStatus` import** (line 31)

2. **Fixed `get_run()` coercion** (line 334):
   ```python
   # Before: export_status=row.export_status
   # After:
   export_status=ExportStatus(row.export_status) if row.export_status else None
   ```

3. **Fixed `list_runs()` coercion** (line 374): Same pattern

4. **Fixed `set_export_status()` method** (lines 383-430):
   - Changed signature: `status: str` → `status: ExportStatus | str`
   - Added validation via `_coerce_enum(status, ExportStatus)`
   - Clear `export_error` when transitioning to `COMPLETED` or `PENDING`
   - Uses `status_enum.value` for DB storage

**Tests added (`tests/core/landscape/test_recorder.py`):**
- `TestExportStatusEnumCoercion` class with 6 regression tests

### Verification

```bash
.venv/bin/python -m pytest tests/core/landscape/test_recorder.py -v
# 97 passed (91 existing + 6 new)
```

### Notes

This fix enforces the Data Manifesto "Tier 1: Full Trust" principle - audit database fields must be properly typed enums, not raw strings. Invalid values will now crash at read time (desired behavior).
