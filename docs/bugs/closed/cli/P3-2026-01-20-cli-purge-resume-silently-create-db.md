# Bug Report: `purge`/`resume` can silently create a new empty Landscape DB on typoed `--database` paths

**Status: CLOSED (Fixed 2026-01-29)**

## Resolution

**Root Cause:** Both `purge` and `resume` bypassed the `resolve_database_url()` helper (which checks file existence) and built the DB URL directly without validation. Combined with `LandscapeDB.from_url()` defaulting to `create_tables=True`, this silently created empty databases.

**Fix Applied:**
1. Added `db_path.exists()` check to both `purge` and `resume` commands in `cli.py`
2. Commands now fail with clear error: "Database file not found: /path/to/db"
3. No database file is created on typoed paths
4. Added explicit tests for missing database behavior
5. Updated existing tests to pre-create databases (they relied on auto-create)

**Files Changed:**
- `src/elspeth/cli.py` - Added existence checks for `--database` paths
- `tests/cli/test_cli.py` - Added 2 new tests, updated 3 existing tests

## Summary

- `elspeth purge` and `elspeth resume` accept `--database` as a SQLite database file path, but do not check that the file exists.
- `LandscapeDB.from_url(..., create_tables=True)` will create the SQLite file (and schema) if it does not exist.
- This can silently create a brand-new empty audit DB when an operator mistypes the path, leading to confusing “nothing to purge” / “run not found” results and polluting the filesystem.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 1 (CLI), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/cli.py` and `src/elspeth/core/landscape/database.py`

## Steps To Reproduce

1. Choose a database path that does not exist, e.g. `/tmp/typo-landscape.db`.
2. Run `elspeth purge --dry-run --database /tmp/typo-landscape.db`.
3. Observe the file now exists on disk and the command reports nothing to delete.

Similar repro:
1. Run `elspeth resume some-run-id --database /tmp/typo-landscape.db`.
2. Observe the file now exists even though no real database was present.

## Expected Behavior

- For destructive/maintenance commands (`purge`, `resume`), `--database` should point to an existing Landscape DB file unless the user explicitly requests initialization.
- On missing DB path, the CLI should error out with a clear message.

## Actual Behavior

- A typoed or wrong path can create a new empty DB, making the command appear to “work” while operating on the wrong data store.

## Evidence

- CLI constructs a sqlite URL from the path and calls `LandscapeDB.from_url(...)`:
  - `src/elspeth/cli.py:514-552` (purge)
  - `src/elspeth/cli.py:632-662` (resume)
- `LandscapeDB.from_url()` defaults to `create_tables=True` and calls `metadata.create_all(engine)`:
  - `src/elspeth/core/landscape/database.py:114-134`

## Impact

- User-facing impact: confusing results; the wrong database gets created/queried.
- Data integrity / security impact: low (does not corrupt the real DB), but can cause operators to miss needed purges/resumes.
- Performance or cost impact: filesystem clutter; potential wasted operator time.

## Root Cause Hypothesis

- `LandscapeDB.from_url()` is designed for “connect or initialize” usage; the CLI uses it unconditionally for commands that should target existing databases.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/cli.py`:
    - For `--database` paths, check `Path(database).expanduser().exists()` and fail if missing.
    - Consider passing `create_tables=False` when connecting for purge/resume.
    - If “create if missing” is desired, add an explicit `--init-db` flag.
- Config or schema changes: none.
- Tests to add/update:
  - Update/add tests ensuring `purge` and `resume` error when `--database` points to a missing file (unless `--init-db` is set).
- Risks or migration steps:
  - This is a behavior change; existing tests currently rely on the “auto-create” behavior for `purge --dry-run`.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: maintenance commands should be conservative with side effects.
- Reason (if known): reusing the same DB constructor as the run path.
- Alignment plan or decision needed: decide intended UX for missing DB files.

## Acceptance Criteria

- `elspeth purge --database /path/does-not-exist.db` exits non-zero and does not create a new DB file.
- `elspeth resume ... --database /path/does-not-exist.db` exits non-zero and does not create a new DB file.

## Tests

- Suggested tests to run:
  - `pytest tests/cli/test_cli.py -k 'purge or resume'`
- New tests required: yes (missing DB path behavior)

## Notes / Links

- Related issues/PRs: N/A

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P3 verification wave 2

**Current Code Analysis:**

The bug is confirmed to still exist in the current codebase. Both `purge` and `resume` commands call `LandscapeDB.from_url(db_url)` without checking if the database file exists first.

**Current code locations:**
- `purge` command: `/home/john/elspeth-rapid/src/elspeth/cli.py:932-1088` (line 1043 calls `LandscapeDB.from_url(db_url)`)
- `resume` command: `/home/john/elspeth-rapid/src/elspeth/cli.py:1219-1405` (line 1291 calls `LandscapeDB.from_url(db_url)`)

**Reproduction confirmed:**

```bash
# Purge command creates empty database
$ rm -f /tmp/typo-landscape.db
$ elspeth purge --dry-run --database /tmp/typo-landscape.db
No payloads older than 90 days found.
$ ls -la /tmp/typo-landscape.db
-rw-r--r-- 1 john john 299008 Jan 25 02:02 /tmp/typo-landscape.db

# Resume command creates empty database
$ rm -f /tmp/typo-resume.db
$ elspeth resume some-fake-run --database /tmp/typo-resume.db
Cannot resume run some-fake-run: Run some-fake-run not found
$ ls -la /tmp/typo-resume.db
-rw-r--r-- 1 john john 299008 Jan 25 02:02 /tmp/typo-resume.db
```

Both commands silently create a 299KB SQLite database file with empty tables when the specified path doesn't exist.

**Git History:**

No commits since 2026-01-20 have addressed this issue. The commands have been refactored (line numbers changed from the original bug report), but the underlying behavior remains the same. Related commits focused on other aspects:
- `bea9bba` - Added EventBus progress reporting to resume
- `879d8ac` - Updated resume to use from_plugin_instances
- Various other resume/checkpoint fixes, but none addressed the silent DB creation issue

**Root Cause Confirmed:**

`LandscapeDB.from_url()` defaults to `create_tables=True` and always creates the database file if it doesn't exist. Neither `purge` nor `resume` pass `create_tables=False` or check for file existence before calling `from_url()`.

The current code at line 1043 (purge) and line 1291 (resume):
```python
db = LandscapeDB.from_url(db_url)
```

This calls `from_url()` with default `create_tables=True`, which creates both the file and schema via `metadata.create_all(engine)` at line 229 of `/home/john/elspeth-rapid/src/elspeth/core/landscape/database.py`.

**Recommendation:**

Keep open. This is a valid UX bug that can confuse operators:

1. A typo in `--database` path creates a new empty DB instead of failing
2. The command appears to succeed but operates on the wrong database
3. Operators may miss needed purges/resumes on the actual database
4. Filesystem pollution with 299KB files in unexpected locations

The fix should check `Path(database).exists()` before calling `from_url()` for maintenance commands, or pass `create_tables=False` and handle the resulting error with a clear message about the missing database.
