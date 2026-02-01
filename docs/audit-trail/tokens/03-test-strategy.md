# Token Outcome Test Strategy

This strategy ensures every token path is covered and every outcome is
recorded correctly. Use a test pyramid: many fast unit tests, fewer
integration tests, and a small number of end-to-end checks.

## Test pyramid targets

- Unit: ~70 percent
- Integration: ~20 percent
- End-to-end: ~10 percent

## Outcome invariants (must hold)

1. Exactly one terminal outcome per token.
2. BUFFERED may repeat, but must be followed by a terminal outcome.
3. Required fields present for each outcome type.
4. COMPLETED implies completed sink node_state.
5. Completed sink node_state implies COMPLETED outcome.
6. Fork/expand children must have token_parents entries.

## Unit tests (fast, isolated)

Focus on outcome recording paths in isolation.

Suggested coverage:
- RowProcessor:
  - QUARANTINED on error_sink == discard
  - ROUTED on error sink
  - FORKED and child creation
  - EXPANDED on multi-row output
  - CONSUMED_IN_BATCH and BUFFERED for aggregation modes
  - FAILED on retry exhaustion
- CoalesceExecutor:
  - COALESCED recorded for consumed tokens on merge
  - FAILED recorded on quorum_not_met / incomplete_branches
  - Late arrival recorded as FAILED (via processor path)
- Recorder:
  - Multiple BUFFERED allowed, terminal unique enforced

Existing tests to extend:
- `tests/engine/test_processor_outcomes.py`
- `tests/engine/test_coalesce_executor_audit_gaps.py`

## Integration tests (pipeline-level)

Use minimal pipeline configurations that exercise full paths end to end.
After each run, execute the audit sweep from
`docs/audit-trail/tokens/02-audit-sweep.md`.

Recommended scenarios:
- Gate route to named sink (ROUTED)
- Fork to two branches with coalesce (FORKED + COALESCED)
- Aggregation passthrough (BUFFERED then COMPLETED)
- Aggregation transform mode (CONSUMED_IN_BATCH + EXPANDED children)
- Transform error to error sink (ROUTED) and discard (QUARANTINED)
- Source quarantine (QUARANTINED with sink_name)

## Property-based tests (invariant hunting)

Use Hypothesis to generate varied token flows and assert invariants.
Focus on the invariants above rather than specific examples.

Suggested properties:
- For any sequence of fork/coalesce/expand operations, each token has
  exactly one terminal outcome by the end of the run.
- BUFFERED tokens always resolve to terminal after flush.
- Required fields are present for each outcome.

Implementation approach:
- Use in-memory LandscapeDB.
- Drive RowProcessor and CoalesceExecutor directly.
- Generate small DAG scenarios with randomized order of branch arrivals.
- Keep max examples low on PR (fast), higher in nightly runs.

## End-to-end checks (few, critical)

Run a small number of full pipelines that include:
- External calls mocked
- Aggregation + coalesce
- Error routing

These should be stable and minimal. Their goal is to detect systemic
regressions, not every edge case.

## CI scheduling

- PR: unit + integration + small property-based sample
- Main: full unit + integration + expanded property-based sample
- Nightly: stress property-based (higher examples) + any E2E
