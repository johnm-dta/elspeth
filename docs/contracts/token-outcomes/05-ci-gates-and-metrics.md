# CI Gates And Metrics For Token Outcomes

Current as of 2026-05-20.

Token outcome gates must enforce the ADR-019 two-axis model. They should fail on
missing lifecycle evidence, illegal pairs, required-field gaps, and sink evidence
disagreement.

## Required Gates

1. No token in a terminal run is missing a completed outcome.
2. No token has more than one completed outcome.
3. No non-terminal outcome uses anything except `(NULL, buffered, completed=0)`.
4. No completed outcome uses an illegal `(outcome, path)` pair.
5. No pair-specific required discriminator field is missing.
6. Sink success outcomes and completed sink node states agree.
7. Parent/delegation paths have corresponding child or batch evidence.

## Suggested CI Stages

- PR: focused unit tests, representative integration tests, and bounded property
  tests for changed producers.
- Main: broader unit and integration suite plus audit sweep assertions on
  representative pipeline runs.
- Nightly/release: expanded property profiles and end-to-end example runs with
  token-outcome sweeps.

## Metrics

| Metric | Target |
|--------|--------|
| Missing completed outcomes / total tokens | 0 |
| Duplicate completed outcomes | 0 |
| Illegal pair count | 0 |
| Required-field violations | 0 |
| Sink success mismatch count | 0 |
| Parent/delegation evidence gaps | 0 |

## Enforcement Notes

- Apply completeness checks only after the run is terminal.
- `buffered` rows may exist during an active run; they must resolve before
  completion.
- Treat any non-zero gate result as a blocking audit-integrity failure.
  Tolerating a known non-zero result requires an explicit accepted audit
  limitation in release evidence, not a silent CI skip.
