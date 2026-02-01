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

---

## VERIFICATION: 2026-01-25

**Status:** OBE (Overtaken By Events)

**Verified By:** Claude Code P3 verification wave 2

**Current Code Analysis:**

The bug has been fixed. Current code in `/home/john/elspeth-rapid/src/elspeth/cli.py` shows:

- Line 575 in `_execute_pipeline()`:
  ```python
  return {
      "run_id": result.run_id,
      "status": result.status.value,  # Convert enum to string for TypedDict
      "rows_processed": result.rows_processed,
  }
  ```

- Line 787 in `_execute_pipeline_with_instances()`:
  ```python
  return {
      "run_id": result.run_id,
      "status": result.status.value,  # Convert enum to string for TypedDict
      "rows_processed": result.rows_processed,
  }
  ```

- Line 1401 in `resume()` command:
  ```python
  typer.echo(f"  Status: {result.status.value}")
  ```

All locations now explicitly call `.value` on the `RunStatus` enum to convert it to its string value.

Additionally, all event formatters use `.value` for status display:
- Line 486 (JSON formatter): `"status": event.status.value`
- Line 540, 752 (console formatters): `event.status.value`
- Line 698 (JSON formatter): `"status": event.status.value`

**Git History:**

The bug was fixed in commit `80c9ae1` on 2026-01-21:

```
commit 80c9ae198fdb8ca5020ca3ffe3a6f82d0f00e8e7
Author: John Morrissey <544926+tachyon-beep@users.noreply.github.com>
Date:   Wed Jan 21 08:02:27 2026 +1100

    fix(contracts): resolve 8 bugs found in contract code review

    CLI:
    - Convert RunStatus enum to string in ExecutionResult (cli.py:367)
    - Use NotRequired for optional TypedDict fields instead of total=False
```

The diff shows:
```diff
-        "status": result.status,
+        "status": result.status.value,  # Convert enum to string for TypedDict
```

This was part of a broader contract code review that fixed 8 bugs.

**Root Cause Confirmed:**

The original bug was exactly as described: `RunStatus(str, Enum)` inherits from both `str` and `Enum`. While this makes it work correctly with database storage and equality checks (the `str` part), the `Enum.__str__()` method formats it as `RunStatus.COMPLETED` when converted to string or used in f-strings.

The fix correctly calls `.value` to extract the string value (`"completed"`) from the enum.

**Test Coverage:**

The fix is indirectly validated by:
- `tests/cli/test_execution_result.py`: Tests that `ExecutionResult` accepts `"completed"` as a string (lines 13-14, 23-24)
- All event formatters consistently use `.value` throughout the codebase

**Recommendation:**

Close as OBE (Overtaken By Events). Bug was fixed on 2026-01-21, one day after it was reported, as part of commit 80c9ae1. The fix is complete, correct, and includes an explanatory comment.
