# Bug Report: `elspeth run` prints `RunStatus.COMPLETED` instead of `completed`

## Summary

- The engine returns `RunResult.status` as a `RunStatus` enum (`class RunStatus(str, Enum)`), but the CLI prints it directly.
- Result: user-visible output shows `RunStatus.COMPLETED` (enum label) rather than the actual status value `completed`.

## Severity

- Severity: trivial
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
- Notable tool calls or steps: code inspection + enum behavior check in Python REPL

## Steps To Reproduce

1. Run any successful pipeline via CLI:
   - `elspeth run -s ./settings.yaml --execute`
2. Observe output:
   - `Run completed: RunStatus.COMPLETED` (or FAILED, etc.)

Minimal enum repro:
1. `python -c "from elspeth.contracts import RunStatus; print(str(RunStatus.COMPLETED)); print(RunStatus.COMPLETED.value)"`
2. Observe `str(...)` prints `RunStatus.COMPLETED` but `.value` is `completed`.

## Expected Behavior

- CLI prints human-readable status values (`completed`, `failed`, `running`), and `ExecutionResult['status']` is a plain string.

## Actual Behavior

- CLI prints enum labels, which is confusing and inconsistent with other output.

## Evidence

- CLI prints status directly:
  - `src/elspeth/cli.py:136`
- CLI returns enum instance as `"status"` in the result dict:
  - `src/elspeth/cli.py:338-342`
- Enum definition:
  - `src/elspeth/contracts/enums.py:11-20`

## Impact

- User-facing impact: confusing output (`RunStatus.COMPLETED`) in the most common CLI success path.
- Data integrity / security impact: none.
- Performance or cost impact: none.

## Root Cause Hypothesis

- `RunStatus` inherits from `(str, Enum)`, so equality/DB storage work, but `Enum.__str__()` formats as `RunStatus.COMPLETED`. The CLI uses f-strings which call `__str__`.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/cli.py`: convert `RunStatus` to its value when constructing `ExecutionResult`:
    - `status = result.status.value` (or `str(result.status.value)`)
  - Alternatively: update print to use `.value` if the dict carries the enum.
- Config or schema changes: none.
- Tests to add/update:
  - Add a CLI test asserting output contains `completed` and not `RunStatus.` on success.
- Risks or migration steps:
  - None.

## Architectural Deviations

- Spec or doc reference: N/A
- Observed divergence: N/A (output formatting bug)
- Reason (if known): N/A
- Alignment plan or decision needed: N/A

## Acceptance Criteria

- Successful `elspeth run` output prints `Run completed: completed`.
- `ExecutionResult["status"]` is a plain string.

## Tests

- Suggested tests to run:
  - `pytest tests/cli/test_cli.py -k run`
- New tests required: yes (output assertion)

## Notes / Links

- Related issues/PRs: N/A
