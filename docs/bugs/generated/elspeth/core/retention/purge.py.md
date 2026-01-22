# Bug Report: Reproducibility grade not updated after payload purge

## Summary

- Purging payloads via `PurgeManager.purge_payloads()` deletes blobs but never updates `runs.reproducibility_grade`, so runs with nondeterministic calls remain marked `REPLAY_REPRODUCIBLE` after payloads are removed, overstating replay capability and violating documented retention semantics.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `/home/john/elspeth-rapid/src/elspeth/core/retention/purge.py`
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): Read-only filesystem sandbox; approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Inspected `src/elspeth/core/retention/purge.py`, `src/elspeth/core/landscape/reproducibility.py`, and `docs/design/architecture.md`

## Steps To Reproduce

1. Create a completed run containing nondeterministic calls so its `reproducibility_grade` is `REPLAY_REPRODUCIBLE`.
2. Ensure the run’s `completed_at` is older than the retention cutoff and its payload refs exist in the payload store.
3. Run `PurgeManager.find_expired_payload_refs()` followed by `PurgeManager.purge_payloads()` for those refs.
4. Query `runs.reproducibility_grade` for the run.

## Expected Behavior

- Purging payloads should degrade the run’s `reproducibility_grade` to `ATTRIBUTABLE_ONLY` once its payloads are removed.

## Actual Behavior

- Purge deletes payloads but leaves `reproducibility_grade` unchanged (e.g., `REPLAY_REPRODUCIBLE`).

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/retention/purge.py:273`, `src/elspeth/core/landscape/reproducibility.py:92`, `docs/design/architecture.md:672`
- Minimal repro input (attach or link): Unknown

## Impact

- User-facing impact: `explain()` and audit views report replay capability that no longer exists after purge.
- Data integrity / security impact: Audit metadata misrepresents reproducibility state, undermining audit assurances.
- Performance or cost impact: Unknown

## Root Cause Hypothesis

- `PurgeManager.purge_payloads()` never invokes `update_grade_after_purge` or tracks affected run IDs, so grades are not degraded when payloads are deleted.

## Proposed Fix

- Code changes (modules/files): Update `src/elspeth/core/retention/purge.py` to track run IDs whose payload refs were actually deleted and call `update_grade_after_purge(self._db, run_id)` for each affected run.
- Config or schema changes: Unknown
- Tests to add/update: Add a retention test that purges expired payloads for a `REPLAY_REPRODUCIBLE` run and asserts `runs.reproducibility_grade` becomes `ATTRIBUTABLE_ONLY`.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md:672`
- Observed divergence: Grade does not degrade after payload purge.
- Reason (if known): Missing integration in `PurgeManager`.
- Alignment plan or decision needed: Call `update_grade_after_purge` as part of purge flow once affected runs are identified.

## Acceptance Criteria

- After purging payloads for a run with nondeterministic calls, `runs.reproducibility_grade` is set to `ATTRIBUTABLE_ONLY` and remains unchanged for runs without purged payloads.

## Tests

- Suggested tests to run: `pytest tests/core/retention/test_purge.py`
- New tests required: Yes, integration test covering purge-induced grade degradation.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `docs/design/architecture.md`
