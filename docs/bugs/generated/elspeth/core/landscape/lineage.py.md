# Bug Report: explain(row_id) returns arbitrary token when multiple tokens exist

## Summary

- `explain()` in `src/elspeth/core/landscape/lineage.py` resolves `row_id` by blindly picking the first token for that row, which yields incomplete or wrong lineage whenever multiple tokens share the same `row_id` (fork/expand/coalesce/resume); this violates the documented requirement to disambiguate by sink and undermines audit accuracy.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (static analysis)
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for `src/elspeth/core/landscape/lineage.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): Read-only sandbox; approvals disabled
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Read `src/elspeth/core/landscape/lineage.py`, `src/elspeth/core/landscape/recorder.py`, `docs/design/architecture.md`

## Steps To Reproduce

1. Create a run and a source row, then fork the token into multiple branches (e.g., via `LandscapeRecorder.fork_token(...)` in a DAG with a gate).
2. Call `explain(recorder, run_id=..., row_id=...)` without a token ID.
3. Observe that the returned lineage corresponds to the first created token, not the terminal token(s) for the row.

## Expected Behavior

- If multiple tokens exist for a `row_id`, `explain()` should either require a sink for disambiguation or select the unique terminal token; it should not arbitrarily pick the first token.

## Actual Behavior

- `explain()` always selects `tokens[0]` when `row_id` is provided, returning lineage for an arbitrary/non-terminal token.

## Evidence

- `src/elspeth/core/landscape/lineage.py:75` documents that `row_id` uses the “first token,” and `src/elspeth/core/landscape/lineage.py:86` selects `tokens[0]`.
- `src/elspeth/core/landscape/recorder.py:793` shows `fork_token` creates multiple tokens for the same `row_id`, making ambiguity routine.
- `docs/design/architecture.md:346` specifies `explain(run_id, row_id, sink, field)` for disambiguation, and `docs/design/architecture.md:347` says row-only is valid only when the row has a single terminal path.

## Impact

- User-facing impact: Auditors/users can receive incomplete or incorrect lineage for rows that fork/expand or are resumed, leading to misleading explanations.
- Data integrity / security impact: Audit integrity is compromised because lineage output can omit actual terminal processing paths.
- Performance or cost impact: None direct.

## Root Cause Hypothesis

- `explain()` resolves `row_id` by taking the first token in creation order instead of validating terminal uniqueness or using sink-based disambiguation.

## Proposed Fix

- Code changes (modules/files): Update `src/elspeth/core/landscape/lineage.py` to accept `sink: str | None`, and when `row_id` is used:
  - If `sink` provided, choose the token whose terminal outcome has `sink_name == sink`.
  - If `sink` not provided, assert there is exactly one terminal token; otherwise return `None` or raise `ValueError`.
- Config or schema changes: None.
- Tests to add/update: Add tests for row_id ambiguity with fork/expand and for row_id+sink disambiguation.
- Risks or migration steps: API signature change for `explain()`; update callers/tests accordingly.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md:341`
- Observed divergence: Implementation lacks sink disambiguation and accepts ambiguous `row_id` queries.
- Reason (if known): Unknown
- Alignment plan or decision needed: Implement sink-based token selection and enforce single-terminal-path rule for row-only queries.

## Acceptance Criteria

- `explain(run_id, row_id=...)` returns lineage only when exactly one terminal token exists; otherwise it errors or returns None.
- `explain(run_id, row_id=..., sink=...)` returns the token whose terminal outcome matches the sink.
- Existing token_id-based queries remain unchanged.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_lineage.py`
- New tests required: Yes — add tests covering row_id ambiguity with fork/expand and row_id+sink disambiguation.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `docs/design/architecture.md:341`
