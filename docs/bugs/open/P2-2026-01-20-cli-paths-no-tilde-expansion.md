# Bug Report: CLI path options do not expand `~` (tilde) / `$HOME`

## Summary

- `elspeth run`, `elspeth validate`, `elspeth purge`, and `elspeth resume` treat user-provided paths as literal strings (no `~` expansion), so common invocations like `-s ~/settings.yaml` fail with “file not found” even when the file exists.
- Affects `--settings`, `--database`, and `--payload-dir`.

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
- Notable tool calls or steps: code inspection of `src/elspeth/cli.py`

## Steps To Reproduce

1. Create a valid settings file at `~/settings.yaml` (or any file under `$HOME`).
2. Run `elspeth validate -s ~/settings.yaml`.
3. Observe `Error: Settings file not found: ~/settings.yaml` even though the file exists.

Similar repros:
- `elspeth purge --database ~/landscape.db`
- `elspeth purge --payload-dir ~/.elspeth/payloads --database ./landscape.db`
- `elspeth resume run-123 --database ~/landscape.db`

## Expected Behavior

- `~` in CLI path arguments should resolve to `$HOME` (standard shell/CLI convention), or the CLI should explicitly document that it does not.

## Actual Behavior

- Paths are passed through `pathlib.Path(...)` without `expanduser()`, so `~` is treated as a literal directory name and file existence checks fail.

## Evidence

- `--settings` uses `Path(settings)` (no `expanduser()`):
  - `src/elspeth/cli.py:82`
  - `src/elspeth/cli.py:355`
- `--database` uses `Path(database)` (no `expanduser()`):
  - `src/elspeth/cli.py:515`
  - `src/elspeth/cli.py:633`
- `--payload-dir` uses `Path(payload_dir)` (no `expanduser()`):
  - `src/elspeth/cli.py:538`

## Impact

- User-facing impact: common invocations fail unexpectedly; operators must remember to manually expand `~` or avoid it.
- Data integrity / security impact: none.
- Performance or cost impact: none.

## Root Cause Hypothesis

- CLI converts strings to `Path` but does not apply `expanduser()` (and uses `str` parameters instead of `Path` parameters with Typer/Click’s path handling).

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/cli.py`: apply `Path(...).expanduser()` for `--settings`, `--database`, and `--payload-dir`.
  - Consider switching option types to `Path` and using Typer’s path validation options (exists/readable/dir_okay/file_okay).
- Config or schema changes: none.
- Tests to add/update:
  - Add CLI tests that create a temp file under `Path.home()` and invoke `validate`/`purge`/`resume` with `~`-prefixed paths.
- Risks or migration steps:
  - Minimal; path expansion is additive and expected.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: N/A (usability/ergonomics bug)
- Reason (if known): N/A
- Alignment plan or decision needed: N/A

## Acceptance Criteria

- `elspeth validate -s ~/settings.yaml` succeeds when the file exists.
- `elspeth purge --database ~/landscape.db` and `elspeth purge --payload-dir ~/.elspeth/payloads ...` correctly resolve the home directory.
- `elspeth resume ... --database ~/landscape.db` correctly resolves the home directory.

## Tests

- Suggested tests to run:
  - `pytest tests/cli/test_cli.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
