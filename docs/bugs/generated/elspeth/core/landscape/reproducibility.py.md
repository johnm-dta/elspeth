# Bug Report: compute_grade ignores invalid determinism values

## Summary

- `compute_grade` only checks for known non-reproducible determinism values and otherwise returns `FULL_REPRODUCIBLE`, so invalid determinism strings in `nodes` are silently treated as reproducible instead of crashing, which violates the Tier 1 audit integrity rule and can mislabel runs.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22T10:48:03+11:00
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `src/elspeth/core/landscape/reproducibility.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Inspected `src/elspeth/core/landscape/reproducibility.py`, `src/elspeth/core/landscape/recorder.py`, `CLAUDE.md`

## Steps To Reproduce

1. Create an in-memory `LandscapeDB` and `LandscapeRecorder`, begin a run, and register a node with valid determinism.
2. Manually corrupt the node determinism with `UPDATE nodes SET determinism='garbage_value' WHERE run_id=...`.
3. Call `compute_grade(db, run_id)` (or `recorder.compute_reproducibility_grade(run_id)`).
4. Observe the function returns `FULL_REPRODUCIBLE` instead of raising on invalid audit data.

## Expected Behavior

- `compute_grade` should fail fast (raise) when any node determinism value is NULL or not in the `Determinism` enum.

## Actual Behavior

- `compute_grade` returns `FULL_REPRODUCIBLE`, masking invalid determinism values.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/landscape/reproducibility.py:56` `src/elspeth/core/landscape/reproducibility.py:74` `src/elspeth/core/landscape/recorder.py:614`
- Minimal repro input (attach or link): `UPDATE nodes SET determinism='garbage_value' WHERE run_id='...'`

## Impact

- User-facing impact: Runs can be labeled fully reproducible even when audit data is corrupt.
- Data integrity / security impact: Violates Tier 1 "crash on anomaly" requirement; corrupted determinism values go undetected.
- Performance or cost impact: Minimal; small additional validation query.

## Root Cause Hypothesis

- `compute_grade` only checks for the presence of known non-reproducible values and never validates that all determinism values are valid enum members.

## Proposed Fix

- Code changes (modules/files): Add determinism validation in `src/elspeth/core/landscape/reproducibility.py` (query for NULL/invalid determinism values or coerce all values to `Determinism` and raise on failure before computing grade).
- Config or schema changes: None.
- Tests to add/update: Add a test that corrupts `nodes.determinism` and asserts `compute_grade` raises, in `tests/core/landscape/test_recorder.py` or `tests/core/landscape/test_reproducibility.py`.
- Risks or migration steps: Slight extra query cost; otherwise none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:40`
- Observed divergence: `compute_grade` tolerates invalid determinism values instead of crashing.
- Reason (if known): Unknown
- Alignment plan or decision needed: Enforce enum validation during grade computation.

## Acceptance Criteria

- `compute_grade` raises on NULL or invalid determinism values and still returns correct grades for valid values.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_recorder.py::TestReproducibilityGradeComputation`
- New tests required: Yes (invalid determinism value should raise)

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md`
---
# Bug Report: update_grade_after_purge leaves FULL_REPRODUCIBLE after payload purge

## Summary

- `update_grade_after_purge` only downgrades `REPLAY_REPRODUCIBLE`; deterministic runs remain `FULL_REPRODUCIBLE` even after payloads are purged, contradicting the documented definition that `ATTRIBUTABLE_ONLY` applies when payloads are purged or absent, which overstates reproducibility.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22T10:48:03+11:00
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `src/elspeth/core/landscape/reproducibility.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Inspected `src/elspeth/core/landscape/reproducibility.py`, `docs/design/architecture.md`, `docs/plans/completed/2026-01-12-phase5-production-hardening.md`

## Steps To Reproduce

1. Create an in-memory `LandscapeDB` and `LandscapeRecorder`, begin a run, register deterministic nodes, and finalize the run (grade is `FULL_REPRODUCIBLE`).
2. Simulate a purge by calling `update_grade_after_purge(db, run_id)`.
3. Fetch the run and observe the grade remains `FULL_REPRODUCIBLE`.

## Expected Behavior

- After payloads are purged or absent, the run grade should be `ATTRIBUTABLE_ONLY` regardless of prior determinism.

## Actual Behavior

- The grade remains `FULL_REPRODUCIBLE` because only `REPLAY_REPRODUCIBLE` is downgraded.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/landscape/reproducibility.py:95` `src/elspeth/core/landscape/reproducibility.py:130` `docs/design/architecture.md:680` `docs/plans/completed/2026-01-12-phase5-production-hardening.md:116`
- Minimal repro input (attach or link): `update_grade_after_purge(db, run_id)` on a run with `FULL_REPRODUCIBLE`

## Impact

- User-facing impact: Runs can be labeled fully reproducible even though payloads are no longer available.
- Data integrity / security impact: Overstates reproducibility; audit consumers may assume recomputation is possible when it is not.
- Performance or cost impact: None.

## Root Cause Hypothesis

- `update_grade_after_purge` encodes the assumption that deterministic runs do not depend on payload retention, conflicting with documented grade semantics.

## Proposed Fix

- Code changes (modules/files): In `src/elspeth/core/landscape/reproducibility.py`, downgrade both `FULL_REPRODUCIBLE` and `REPLAY_REPRODUCIBLE` to `ATTRIBUTABLE_ONLY` when payloads are purged.
- Config or schema changes: None.
- Tests to add/update: Update tests that assert `FULL_REPRODUCIBLE` remains unchanged after purge in `tests/core/landscape/test_recorder.py`.
- Risks or migration steps: Align on intended semantics; this is a behavior change for existing tests and reports.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md:680`
- Observed divergence: Implementation keeps `FULL_REPRODUCIBLE` after purge despite "payloads purged or absent" mapping to `ATTRIBUTABLE_ONLY`.
- Reason (if known): Unknown
- Alignment plan or decision needed: Confirm whether `FULL_REPRODUCIBLE` should degrade on purge; if yes, update code/tests accordingly.

## Acceptance Criteria

- When payloads are purged or absent, the run grade becomes `ATTRIBUTABLE_ONLY`.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_recorder.py::TestReproducibilityGradeComputation`
- New tests required: Update existing purge-grade tests to match revised semantics.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `docs/design/architecture.md`
