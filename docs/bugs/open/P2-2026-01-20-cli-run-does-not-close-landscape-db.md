# Bug Report: CLI `run` path never closes `LandscapeDB` (SQLite file handles/locks can linger)

## Summary

- `_execute_pipeline()` opens a `LandscapeDB` via `LandscapeDB.from_url(...)` but does not close it.
- Other CLI commands (`purge`, `resume`) correctly close the database in a `finally` block, so `run` is inconsistent.

## Severity

- Severity: minor
- Priority: P2

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

1. Run `elspeth run -s ./settings.yaml --execute` repeatedly in a long-lived Python process that imports and calls `_execute_pipeline()` directly (or invoke CLI via `CliRunner` without process exit).
2. Observe increasing open file descriptors / SQLite handles until GC/process exit.

## Expected Behavior

- Database connections/engines created by CLI should always be closed/disposed when the command completes (success or failure).

## Actual Behavior

- `_execute_pipeline()` returns without closing the `LandscapeDB`.

## Evidence

- `LandscapeDB.from_url(...)` is called in `_execute_pipeline()`:
  - `src/elspeth/cli.py:274-276`
- No corresponding `db.close()` in `_execute_pipeline()`.
- `purge` closes DB in `finally`:
  - `src/elspeth/cli.py:596-597`
- `resume` closes DB in `finally`:
  - `src/elspeth/cli.py:696-697`
- `LandscapeDB.close()` disposes the engine:
  - `src/elspeth/core/landscape/database.py:79-83`

## Impact

- User-facing impact: potential “database is locked” behavior in embedded contexts; file handles linger until process exit.
- Data integrity / security impact: low (primarily operational correctness).
- Performance or cost impact: resource leakage in long-running processes/tests.

## Root Cause Hypothesis

- `_execute_pipeline()` constructs the DB as a local variable without `try/finally` or a context manager.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/cli.py`:
    - Wrap DB usage in `try/finally: db.close()`, or use `with LandscapeDB.from_url(db_url) as db:`.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that invokes `_execute_pipeline()` (or CLI run via `CliRunner`) and then asserts the engine is disposed (or that the DB file can be safely removed on Windows-like semantics).
- Risks or migration steps:
  - None.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: resource lifecycle is inconsistent across CLI commands.
- Reason (if known): N/A
- Alignment plan or decision needed: N/A

## Acceptance Criteria

- `LandscapeDB.close()` is always called after `elspeth run` (success or failure).
- No resource leakage in repeated calls from tests/embedded environments.

## Tests

- Suggested tests to run:
  - `pytest tests/cli/test_cli.py -k run`
- New tests required: yes (resource lifecycle)

## Notes / Links

- Related issues/PRs: N/A
