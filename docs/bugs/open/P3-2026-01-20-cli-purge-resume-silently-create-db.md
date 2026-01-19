# Bug Report: `purge`/`resume` can silently create a new empty Landscape DB on typoed `--database` paths

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
